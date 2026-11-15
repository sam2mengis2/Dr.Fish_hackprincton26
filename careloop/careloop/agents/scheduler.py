"""Agent 11 - Scheduler (plain code, one mock calendar per hospital).

Leyat's pain point: patients wait 5-6 months for CT/MRI. CareLoop can't add scanners, but it can
  1. book the earliest slot across every hospital in the network (shortest line), preferring the
     patient's own hospital only when that still meets the clinical due date;
  2. put anyone whose slot is later than their guideline due date on the cancellation list;
  3. when a slot frees up (cancellation, reschedule, an earlier offer accepted), offer it by text to the
     waiting patient with the most urgent clinical due date ("reply TAKE"); if they skip or don't answer
     in time, the next patient gets it. Each accepted offer frees a later slot, which is offered on in turn.
Backlogs per site come from config scheduling.backlog_days (empty = no backlog, used by tests/evals).
Swap MockCalendar for each hospital's scheduling API later.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import re

from ..knowledge import is_clinician_task, patient_label, test_type
from ..runtime import Runtime
from ..schema import Case, Message, Timer

SLOT_HOURS = [9, 10, 11, 13, 14, 15]
NOT_BOOKABLE = ("review", "cultures", "cbc", "crp", "bnp")
DEFAULT_SITE = "Main imaging"


class MockCalendar:
    def __init__(self):
        self.booked: set[str] = set()

    def peek(self, earliest: datetime) -> datetime:
        d = earliest.date()
        while True:
            if d.weekday() < 5:
                for h in SLOT_HOURS:
                    slot = datetime.combine(d, time(h))
                    if slot >= earliest and slot.isoformat(timespec="minutes") not in self.booked:
                        return slot
            d += timedelta(days=1)

    def take(self, slot: datetime | str) -> None:
        self.booked.add(slot if isinstance(slot, str) else slot.isoformat(timespec="minutes"))

    def release(self, slot: str) -> None:
        self.booked.discard(slot)

    def first_free(self, earliest: datetime) -> datetime:
        slot = self.peek(earliest)
        self.take(slot)
        return slot


SCANNERS = {"ct-chest": "CT", "pet-ct": "PET-CT", "mri-brain": "MRI"}


def resource(action: str) -> str:
    """Which queue an appointment waits in: a scanner (CT / MRI / PET-CT) or a clinic visit.
    Scanner backlogs never apply to clinic visits, and a freed CT slot only goes to someone waiting for a CT."""
    kind = test_type(action)
    if kind in SCANNERS:
        return SCANNERS[kind]
    a = action.lower()
    if re.search(r"\bpet\b", a):
        return "PET-CT"
    if re.search(r"\bmri\b|magnetic resonance", a):
        return "MRI"
    if re.search(r"\bct\b|computed tomography", a):
        return "CT"
    return "clinic"


def _calendar(rt: Runtime, site: str = DEFAULT_SITE, res: str = "clinic") -> MockCalendar:
    if not hasattr(rt, "calendars"):
        rt.calendars = {}
    site = site or DEFAULT_SITE
    key = (site, res)
    if key not in rt.calendars:  # rebuild from stored bookings (API restarts keep their calendars)
        cal = MockCalendar()
        for c in rt.store.list():
            for f in c.followups:
                if resource(f.action) != res:
                    continue
                if f.site == site and f.appointment and f.status in ("booked", "missed"):
                    cal.take(f.appointment)
                if f.offer and f.offer.get("site") == site:
                    cal.take(f.offer["slot"])
        rt.calendars[key] = cal
    return rt.calendars[key]


def _sched(rt: Runtime) -> dict:
    return rt.config.get("scheduling") or {}


def _backlog_start(rt: Runtime, site: str, res: str = "CT") -> datetime:
    days = 0 if res == "clinic" else (_sched(rt).get("backlog_days") or {}).get(site, 0)  # scanners only
    return (rt.clock.now() + timedelta(days=days)).replace(hour=8, minute=0, second=0, microsecond=0)


def sites(case: Case, rt: Runtime | None = None) -> list[str]:
    """Hospitals this patient can be booked at: preferred site first, then wherever they've been scanned,
    then (if a runtime is given) the rest of the network."""
    out = []
    network = list((_sched(rt).get("backlog_days") or {}).keys()) if rt else []
    for s in [case.patient.preferred_site] + [sc.site for sc in reversed(case.scans)] + network:
        if s and s not in out:
            out.append(s)
    return out or [DEFAULT_SITE]


def _when(slot: datetime | str) -> str:
    slot = datetime.fromisoformat(slot) if isinstance(slot, str) else slot
    return slot.strftime("%a %b %d at %I:%M %p")


def _tell(case: Case, rt: Runtime, msg: str) -> None:
    rt.messenger.send("patient", case.patient.patient_id, msg)
    case.messages.append(Message(to="patient", text=msg, at=rt.clock.iso()))


def _set_timers(case: Case, fu_id: str, slot: datetime, rt: Runtime) -> None:
    case.timers = [t for t in case.timers if not (t.ref == fu_id and t.kind in ("appt_reminder", "appt_missed_check"))]
    for days in rt.config["timers"]["appt_reminder_days_before"]:
        at = slot - timedelta(days=days)
        if at > rt.clock.now():
            case.timers.append(Timer(kind="appt_reminder", fire_at=at.isoformat(timespec="minutes"), ref=fu_id))
    case.timers.append(Timer(kind="appt_missed_check", fire_at=(slot + timedelta(hours=2)).isoformat(timespec="minutes"),
                             ref=fu_id))


def _pick_slot(case: Case, rt: Runtime, earliest: datetime, due: datetime, res: str) -> tuple[str, datetime, str]:
    """Preferred hospital if it meets the due date; otherwise the earliest slot anywhere in the network."""
    options = []
    for site in sites(case, rt):
        start = max(earliest, _backlog_start(rt, site, res))
        options.append((site, _calendar(rt, site, res).peek(start)))
    pref_site, pref_slot = options[0]
    if pref_slot <= due or len(options) == 1:
        note = "patient's preferred hospital" if case.patient.preferred_site == pref_site else ""
        return pref_site, pref_slot, note
    site, slot = min(options, key=lambda o: o[1])
    note = f"earliest of {len(options)} hospitals"
    if site != pref_site:
        note += f" ({pref_site} next free {pref_slot.date().isoformat()})"
    return site, slot, note


def book_all(case: Case, rt: Runtime) -> Case:
    """Books items in plan order, at least one day apart, so later steps (e.g. biopsy) follow earlier
    results (e.g. PET-CT) instead of everything landing on the same morning."""
    now = rt.clock.now()
    existing = [datetime.fromisoformat(f.appointment) for f in case.followups
                if f.appointment and f.status in ("booked", "missed")]
    todo = sorted((fu for fu in case.followups
                   if fu.status in ("open", "patient_notified") and not fu.appointment
                   and not any(k in fu.action.lower() for k in NOT_BOOKABLE)
                   and not is_clinician_task(fu.action)), key=lambda f: f.due)
    for fu in todo:
        due = datetime.fromisoformat(fu.due + "T17:00")
        earliest = max(now + timedelta(days=1), due - timedelta(days=7)).replace(hour=8, minute=0)
        before_due = [e for e in existing if e < due]
        if before_due:
            chained = (max(before_due) + timedelta(days=1)).replace(hour=8, minute=0)
            if chained <= due:
                earliest = max(earliest, chained)
        res = resource(fu.action)
        if fu.site:
            site = fu.site
            slot = _calendar(rt, site, res).peek(max(earliest, _backlog_start(rt, site, res)))
            note = fu.booking_note
        else:
            site, slot, note = _pick_slot(case, rt, earliest, due, res)
        _calendar(rt, site, res).take(slot)
        existing.append(slot)
        fu.appointment, fu.site, fu.status = slot.isoformat(timespec="minutes"), site, "booked"
        fu.waitlisted = slot > due
        fu.booking_note = note + ("; later than due date -> on cancellation list" if fu.waitlisted else "")
        _set_timers(case, fu.id, slot, rt)
        case.log("scheduler", f"{fu.id} '{fu.action}' ({res}) booked {fu.appointment} at {site} (due {fu.due})"
                              + (f" [{fu.booking_note}]" if fu.booking_note else ""), rt.clock.iso())
    return case


# ----------------------------------------------------------------------------- freeing and offering slots
def release(rt: Runtime, site: str, slot: str, res: str, exclude_pid: str | None = None) -> str | None:
    """A slot is free again: give it to whoever needs it most. Returns the patient it was offered to."""
    _calendar(rt, site, res).release(slot)
    return backfill(rt, site, slot, res, exclude_pid=exclude_pid)


def backfill(rt: Runtime, site: str, slot: str, res: str, exclude_pid: str | None = None) -> str | None:
    s = _sched(rt)
    when = datetime.fromisoformat(slot)
    if when < rt.clock.now() + timedelta(hours=s.get("min_lead_hours", 24)):
        return None  # too soon to fill by text; the hospital's own same-day list handles it
    cands = []
    for c in rt.store.list():
        if c.patient.patient_id == exclude_pid or site not in sites(c, rt):
            continue
        for f in c.followups:
            if f.status == "booked" and f.appointment and f.appointment[:10] > slot[:10] and not f.offer \
                    and slot not in f.declined and resource(f.action) == res:
                cands.append((f.due, f.appointment, c, f))
    if not cands:
        return None
    _, _, case, fu = min(cands, key=lambda x: (x[0], x[1]))  # most urgent clinical due date first
    _calendar(rt, site, res).take(slot)  # hold it while the patient decides
    hours = s.get("offer_hours", 4)
    expires = (rt.clock.now() + timedelta(hours=hours)).isoformat(timespec="minutes")
    fu.offer = {"site": site, "slot": slot, "expires": expires, "resource": res}
    case.log("scheduler", f"{fu.id} offered earlier slot {slot} at {site} (due {fu.due}, now {fu.appointment})",
             rt.clock.iso())
    if case.patient.demo_auto_reply:
        accept_offer(case, rt)
    else:
        case.timers.append(Timer(kind="offer_expire", ref=fu.id, fire_at=expires))
        _tell(case, rt, f"Good news: an earlier {patient_label(fu.action)} slot opened at {site} on {_when(slot)} "
                        f"(you're booked {_when(fu.appointment)}). Reply TAKE to grab it or SKIP. "
                        f"We'll hold it for {hours} hours. - your care team")
    rt.store.save(case)
    return case.patient.patient_id


def accept_offer(case: Case, rt: Runtime) -> Case:
    fu = next((f for f in case.followups if f.offer), None)
    if fu is None:
        _tell(case, rt, "That slot isn't available anymore, sorry. We'll text you if another opens. - your care team")
        return case
    old_site, old_slot = fu.site, fu.appointment
    fu.site, fu.appointment, fu.status = fu.offer["site"], fu.offer["slot"], "booked"
    fu.offer = None
    for t in case.timers:
        if t.ref == fu.id and t.kind == "offer_expire":
            t.done = True
    fu.waitlisted = datetime.fromisoformat(fu.appointment) > datetime.fromisoformat(fu.due + "T17:00")
    fu.booking_note = f"moved up from {old_slot[:10]} via cancellation"
    _set_timers(case, fu.id, datetime.fromisoformat(fu.appointment), rt)
    gained = (datetime.fromisoformat(old_slot) - datetime.fromisoformat(fu.appointment)).days
    fu.days_saved += max(gained, 0)
    case.log("scheduler", f"{fu.id} took earlier slot {fu.appointment} at {fu.site} ({gained} days sooner)",
             rt.clock.iso())
    if not case.patient.demo_auto_reply:
        _tell(case, rt, f"Done - your {patient_label(fu.action)} is now {_when(fu.appointment)} at {fu.site}, "
                        f"{gained} days sooner. - your care team")
    rt.store.save(case)
    if old_slot and old_site:
        release(rt, old_site, old_slot, resource(fu.action), exclude_pid=case.patient.patient_id)  # pass it on
    return case


def decline_offer(case: Case, rt: Runtime, expired: bool = False) -> Case:
    fu = next((f for f in case.followups if f.offer), None)
    if fu is None:
        return case
    site, slot = fu.offer["site"], fu.offer["slot"]
    fu.declined.append(slot)
    fu.offer = None
    for t in case.timers:
        if t.ref == fu.id and t.kind == "offer_expire":
            t.done = True
    case.log("scheduler", f"{fu.id} {'did not answer' if expired else 'skipped'} offer {slot} -> next patient",
             rt.clock.iso())
    if not expired:
        _tell(case, rt, f"No problem - you keep {_when(fu.appointment)}. - your care team")
    rt.store.save(case)
    release(rt, site, slot, resource(fu.action), exclude_pid=case.patient.patient_id)
    return case


def cancel(case: Case, rt: Runtime, fu_id: str | None = None, reason: str = "") -> Case:
    """Patient cancels a booking: the slot goes straight to the next patient who needs it."""
    fu = case.followup(fu_id) if fu_id else min(
        (f for f in case.followups if f.status == "booked" and f.appointment),
        key=lambda f: f.appointment, default=None)
    if fu is None or not fu.appointment:
        return case
    site, slot = fu.site, fu.appointment
    fu.status, fu.appointment = "cancelled", None
    for t in case.timers:
        if t.ref == fu.id:
            t.done = True
    case.log("scheduler", f"{fu.id} cancelled by patient ({reason or 'no reason given'}); slot {slot} at {site} freed",
             rt.clock.iso())
    if not case.patient.demo_auto_reply:
        _tell(case, rt, "Your appointment is cancelled. Your care team has been told. - your care team")
        rt.messenger.send("nurse", case.patient.patient_id,
                          f"{case.patient.patient_id} cancelled '{fu.action}' (due {fu.due}). Please confirm with the doctor.")
    rt.store.save(case)
    release(rt, site, slot, resource(fu.action), exclude_pid=case.patient.patient_id)
    return case


# ----------------------------------------------------------------------------- patient-driven changes
def reschedule_next(case: Case, rt: Runtime) -> Case:
    # Prefer the item the patient just missed; otherwise the next upcoming booking.
    fu = next((f for f in case.followups if f.status in ("missed", "escalated")), None) or \
        min((f for f in case.followups if f.status == "booked" and f.appointment),
            key=lambda f: f.appointment, default=None)
    if fu is None:
        return case
    old = datetime.fromisoformat(fu.appointment) if fu.appointment else rt.clock.now()
    old_slot, was_future = fu.appointment, fu.appointment and old > rt.clock.now()
    site = fu.site or sites(case)[0]
    res = resource(fu.action)
    slot = _calendar(rt, site, res).first_free(max(old, rt.clock.now(), _backlog_start(rt, site, res)) + timedelta(days=1))
    fu.appointment = slot.isoformat(timespec="minutes")
    fu.status = "booked"
    _set_timers(case, fu.id, slot, rt)
    where = f" at {fu.site}" if fu.site else ""
    _tell(case, rt, f"Done - your {patient_label(fu.action)} is now {_when(slot)}{where}. - your care team")
    case.log("scheduler", f"{fu.id} rescheduled to {fu.appointment}", rt.clock.iso())
    if was_future:
        rt.store.save(case)
        release(rt, site, old_slot, res, exclude_pid=case.patient.patient_id)
    return case


def switch_site(case: Case, rt: Runtime) -> Case:
    """Patient asked for a different hospital: move the next booking to the other site, keeping the due date."""
    fu = next((f for f in case.followups if f.status in ("missed", "escalated")), None) or \
        min((f for f in case.followups if f.status == "booked" and f.appointment),
            key=lambda f: f.appointment, default=None)
    options = sites(case, rt)
    if fu is None or len(options) < 2:
        _tell(case, rt, "Sorry, we don't have another hospital to offer for this appointment. Reply RESCHEDULE for "
                        "a different time. - your care team")
        return case
    new_site = options[(options.index(fu.site) + 1) % len(options)] if fu.site in options else options[0]
    due = datetime.fromisoformat(fu.due + "T17:00")
    res = resource(fu.action)
    earliest = max(rt.clock.now() + timedelta(days=1), due - timedelta(days=7), _backlog_start(rt, new_site, res))
    slot = _calendar(rt, new_site, res).first_free(earliest.replace(hour=8, minute=0))
    old_site, old_slot = fu.site, fu.appointment
    fu.site, fu.appointment, fu.status = new_site, slot.isoformat(timespec="minutes"), "booked"
    fu.waitlisted = slot > due
    _set_timers(case, fu.id, slot, rt)
    _tell(case, rt, f"Done - your {patient_label(fu.action)} is now at {new_site} on {_when(slot)}. "
                    f"We've let both hospitals know. - your care team")
    case.log("scheduler", f"{fu.id} moved from {old_site} to {new_site}, {fu.appointment}", rt.clock.iso())
    if old_site and old_slot and datetime.fromisoformat(old_slot) > rt.clock.now():
        rt.store.save(case)
        release(rt, old_site, old_slot, res, exclude_pid=case.patient.patient_id)
    return case
