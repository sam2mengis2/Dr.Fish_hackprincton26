"""Agent 12 - Follow-up Watchdog (plain code: timer queue + escalation chain).

tick() fires every due timer on the simulated clock:
  approval_reminder -> nudge the doctor;  approval_escalate -> covering doctor / nurse navigator
  appt_reminder     -> remind the patient
  appt_missed_check -> if the follow-up scan hasn't arrived: mark missed, offer re-book, set escalation
  rebook_escalate   -> still missed: nurse navigator calls the patient
  report_overdue    -> radiologist report not signed in time: alert the reading radiologist
  report_overdue_2  -> still unsigned: alert the radiology lead and tell the ordering doctor it's delayed
A case closes only when every follow-up item is completed (closed by intake when the scan arrives).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..knowledge import patient_label
from ..runtime import Runtime
from ..schema import Case, CaseState, Message, Timer


def _say(case: Case, rt: Runtime, to: str, text: str) -> None:
    rt.messenger.send(to, case.patient.patient_id, text)
    case.messages.append(Message(to=to, text=text, at=rt.clock.iso()))


def tick(case: Case, rt: Runtime) -> Case:
    now = rt.clock.now()
    pid = case.patient.patient_id
    for t in sorted(case.timers, key=lambda x: x.fire_at):
        if t.done or datetime.fromisoformat(t.fire_at) > now:
            continue
        t.done = True
        fu = case.followup(t.ref) if t.ref else None
        waiting = case.state in (CaseState.AWAITING_APPROVAL, CaseState.NEEDS_HUMAN_REVIEW)
        if t.kind == "approval_reminder" and waiting:
            _say(case, rt, "doctor", f"Reminder: CareLoop plan for {pid} is awaiting your reply (APPROVE / EDIT / REJECT / CALL).")
            case.log("watchdog", "approval reminder sent", rt.clock.iso())
        elif t.kind == "approval_escalate" and waiting:
            case.state = CaseState.ESCALATED
            _say(case, rt, "covering_doctor", f"ESCALATION: {case.patient.ordering_doctor} has not responded about {pid} "
                                              f"({case.urgency}). Please review: {case.plan_summary}")
            case.log("watchdog", "approval overdue -> escalated to covering doctor", rt.clock.iso())
        elif t.kind == "appt_reminder" and fu and fu.status == "booked" \
                and datetime.fromisoformat(fu.appointment) > now \
                and now - datetime.fromisoformat(t.fire_at) < timedelta(hours=6):  # skip stale reminders after a clock jump
            when = datetime.fromisoformat(fu.appointment).strftime("%a %b %d at %I:%M %p")
            where = f" at {fu.site}" if fu.site else ""
            _say(case, rt, "patient", f"Reminder: your {patient_label(fu.action)} is on {when}{where}. "
                                      "Reply RESCHEDULE if you need another time.")
            case.log("watchdog", f"{fu.id} reminder sent", rt.clock.iso())
        elif t.kind == "appt_missed_check" and fu and fu.status == "booked":
            fu.status = "missed"
            _say(case, rt, "patient", f"We missed you at your {patient_label(fu.action)}. Reply RESCHEDULE and we'll find a new time.")
            hours = rt.config["timers"]["rebook_escalate_hours"]
            case.timers.append(Timer(kind="rebook_escalate", ref=fu.id,
                                     fire_at=(now + timedelta(hours=hours)).isoformat(timespec="minutes")))
            case.log("watchdog", f"{fu.id} missed -> re-book offer sent", rt.clock.iso())
        elif t.kind == "rebook_escalate" and fu and fu.status == "missed":
            fu.status = "escalated"
            _say(case, rt, "nurse", f"Nurse navigator: {pid} missed '{fu.action}' and hasn't re-booked. Please call the patient.")
            case.log("watchdog", f"{fu.id} still missed -> nurse navigator alerted", rt.clock.iso())
        elif t.kind == "offer_expire" and fu and fu.offer:
            from .scheduler import decline_offer

            _say(case, rt, "patient", "The earlier slot we offered has been passed to the next patient. You keep "
                                      "your current appointment. - your care team")
            decline_offer(case, rt, expired=True)
        elif t.kind in ("report_overdue", "report_overdue_2"):
            scan = next((s for s in case.scans if s.scan_id == t.ref and s.report_status == "pending"), None)
            if scan is None:
                continue
            hrs = (now - datetime.fromisoformat(scan.acquired_at)).total_seconds() / 3600 if scan.acquired_at else 0
            exam = f"{scan.modality} {scan.body_part.lower()}"
            if t.kind == "report_overdue":
                _say(case, rt, "radiologist", f"Report overdue: {pid} {exam} ({scan.read_priority}) unsigned for "
                                              f"{hrs:.0f} h. Order reason: {scan.indication or 'n/a'}.")
                case.log("watchdog", f"{scan.scan_id} report overdue -> radiologist alerted", rt.clock.iso())
            else:
                _say(case, rt, "radiologist", f"SECOND NOTICE to radiology lead: {pid} {exam} still unsigned after "
                                              f"{hrs:.0f} h. Please reassign.")
                _say(case, rt, "doctor", f"FYI: the report for {pid}'s {exam} is delayed ({hrs:.0f} h). Radiology has "
                                         f"been notified; you'll get a text the moment it's signed.")
                case.log("watchdog", f"{scan.scan_id} still unsigned -> radiology lead + ordering doctor told",
                         rt.clock.iso())
    pending_report = any(s.report_status == "pending" for s in case.scans)
    if case.followups and not pending_report and all(f.status == "completed" for f in case.followups):
        if case.state != CaseState.CLOSED:
            case.state = CaseState.CLOSED
            case.log("watchdog", "all follow-ups completed -> case closed", rt.clock.iso())
    return case
