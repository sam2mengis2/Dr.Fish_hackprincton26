"""Agent 11 - Scheduler (plain code, mock clinic calendar).

Books the first free weekday slot on or after max(tomorrow, due - 7 days), writes the
appointment to the follow-up item, and sets reminder + missed-appointment timers.
Swap MockCalendar for the hospital scheduling API later.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

from ..knowledge import patient_label
from ..runtime import Runtime
from ..schema import Case, Timer

SLOT_HOURS = [9, 10, 11, 13, 14, 15]
NOT_BOOKABLE = ("review", "cultures", "cbc", "crp", "bnp")


class MockCalendar:
    def __init__(self):
        self.booked: set[str] = set()

    def first_free(self, earliest: datetime) -> datetime:
        d = earliest.date()
        while True:
            if d.weekday() < 5:
                for h in SLOT_HOURS:
                    slot = datetime.combine(d, time(h))
                    if slot >= earliest and slot.isoformat(timespec="minutes") not in self.booked:
                        self.booked.add(slot.isoformat(timespec="minutes"))
                        return slot
            d += timedelta(days=1)


def _calendar(rt: Runtime) -> MockCalendar:
    if not hasattr(rt, "calendar"):
        rt.calendar = MockCalendar()
    return rt.calendar


def _set_timers(case: Case, fu_id: str, slot: datetime, rt: Runtime) -> None:
    case.timers = [t for t in case.timers if not (t.ref == fu_id and t.kind in ("appt_reminder", "appt_missed_check"))]
    for days in rt.config["timers"]["appt_reminder_days_before"]:
        at = slot - timedelta(days=days)
        if at > rt.clock.now():
            case.timers.append(Timer(kind="appt_reminder", fire_at=at.isoformat(timespec="minutes"), ref=fu_id))
    case.timers.append(Timer(kind="appt_missed_check", fire_at=(slot + timedelta(hours=2)).isoformat(timespec="minutes"),
                             ref=fu_id))


def book_all(case: Case, rt: Runtime) -> Case:
    """Books items in plan order, at least one day apart, so later steps (e.g. biopsy) follow earlier
    results (e.g. PET-CT) instead of everything landing on the same morning."""
    now = rt.clock.now()
    existing = [datetime.fromisoformat(f.appointment) for f in case.followups
                if f.appointment and f.status in ("booked", "missed")]
    last: datetime | None = max(existing) if existing else None
    for fu in case.followups:
        if fu.status not in ("open", "patient_notified") or fu.appointment:
            continue
        if any(k in fu.action.lower() for k in NOT_BOOKABLE):
            continue
        due = datetime.fromisoformat(fu.due + "T17:00")
        earliest = max(now + timedelta(days=1), due - timedelta(days=7)).replace(hour=8, minute=0)
        if last is not None:
            earliest = max(earliest, (last + timedelta(days=1)).replace(hour=8, minute=0))
        slot = _calendar(rt).first_free(earliest)
        last = slot
        fu.appointment = slot.isoformat(timespec="minutes")
        fu.status = "booked"
        _set_timers(case, fu.id, slot, rt)
        case.log("scheduler", f"{fu.id} '{fu.action}' booked {fu.appointment} (due {fu.due})", rt.clock.iso())
    return case


def reschedule_next(case: Case, rt: Runtime) -> Case:
    # Prefer the item the patient just missed; otherwise the next upcoming booking.
    fu = next((f for f in case.followups if f.status in ("missed", "escalated")), None) or \
        min((f for f in case.followups if f.status == "booked" and f.appointment),
            key=lambda f: f.appointment, default=None)
    if fu is None:
        return case
    old = datetime.fromisoformat(fu.appointment) if fu.appointment else rt.clock.now()
    slot = _calendar(rt).first_free(max(old, rt.clock.now()) + timedelta(days=1))
    fu.appointment = slot.isoformat(timespec="minutes")
    fu.status = "booked"
    _set_timers(case, fu.id, slot, rt)
    msg = f"Done - your {patient_label(fu.action)} is now {slot.strftime('%a %b %d at %I:%M %p')}. - your care team"
    rt.messenger.send("patient", case.patient.patient_id, msg)
    from ..schema import Message

    case.messages.append(Message(to="patient", text=msg, at=rt.clock.iso()))
    case.log("scheduler", f"{fu.id} rescheduled to {fu.appointment}", rt.clock.iso())
    return case
