"""Agent 12 - Follow-up Watchdog (plain code: timer queue + escalation chain).

tick() fires every due timer on the simulated clock:
  approval_reminder -> nudge the doctor;  approval_escalate -> covering doctor / nurse navigator
  appt_reminder     -> remind the patient
  appt_missed_check -> if the follow-up scan hasn't arrived: mark missed, offer re-book, set escalation
  rebook_escalate   -> still missed: nurse navigator calls the patient
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
        if t.kind == "approval_reminder" and case.state == CaseState.AWAITING_APPROVAL:
            _say(case, rt, "doctor", f"Reminder: CareLoop plan for {pid} is awaiting your reply (APPROVE / EDIT / REJECT / CALL).")
            case.log("watchdog", "approval reminder sent", rt.clock.iso())
        elif t.kind == "approval_escalate" and case.state == CaseState.AWAITING_APPROVAL:
            case.state = CaseState.ESCALATED
            _say(case, rt, "covering_doctor", f"ESCALATION: {case.patient.ordering_doctor} has not responded about {pid} "
                                              f"({case.urgency}). Please review: {case.plan_summary}")
            case.log("watchdog", "approval overdue -> escalated to covering doctor", rt.clock.iso())
        elif t.kind == "appt_reminder" and fu and fu.status == "booked" \
                and datetime.fromisoformat(fu.appointment) > now \
                and now - datetime.fromisoformat(t.fire_at) < timedelta(hours=6):  # skip stale reminders after a clock jump
            when = datetime.fromisoformat(fu.appointment).strftime("%a %b %d at %I:%M %p")
            _say(case, rt, "patient", f"Reminder: your {patient_label(fu.action)} is on {when}. Reply RESCHEDULE if you need another time.")
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
    if case.followups and all(f.status == "completed" for f in case.followups):
        if case.state != CaseState.CLOSED:
            case.state = CaseState.CLOSED
            case.log("watchdog", "all follow-ups completed -> case closed", rt.clock.iso())
    return case
