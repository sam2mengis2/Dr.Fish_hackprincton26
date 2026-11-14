"""Service layer: the few entry points the API, dashboard, demo and OpenHospital all call."""
from __future__ import annotations

from datetime import datetime

from .agents import clinician_comms, intake, patient_comms, scheduler, watchdog
from .graph import build_graph
from .knowledge import patient_label
from .runtime import Runtime
from .schema import Case, CaseState, Message, Patient, Scan


def submit_scan(rt: Runtime, patient: Patient, scan: Scan) -> Case:
    """A new scan arrives: intake, then run the graph until the doctor has been texted."""
    case = intake.ingest(rt, patient, scan)
    result = build_graph(rt).invoke({"case": case}, {"recursion_limit": 50})
    case = result["case"]
    rt.store.save(case)
    return case


def doctor_reply(rt: Runtime, patient_id: str, text: str) -> tuple[Case, str]:
    case = _get(rt, patient_id)
    case, outcome = clinician_comms.handle_reply(case, text, rt)
    if outcome == "APPROVE":
        case = _activate(case, rt)
    elif outcome == "EDIT":
        # revised plan goes back to the doctor for re-approval
        case = clinician_comms.send_plan(case, rt)
    rt.store.save(case)
    return case, outcome


def patient_reply(rt: Runtime, patient_id: str, text: str) -> tuple[Case, str]:
    case = _get(rt, patient_id)
    case, outcome = patient_comms.handle_reply(case, text, rt)
    rt.store.save(case)
    return case, outcome


def record_result(rt: Runtime, patient_id: str, followup_id: str, note: str = "") -> Case:
    """A non-imaging result arrived (lab value, consult note): close that follow-up item.
    Imaging follow-ups are closed by submit_scan when the actual scan arrives."""
    case = _get(rt, patient_id)
    fu = case.followup(followup_id)
    if fu is None:
        raise KeyError(f"no follow-up {followup_id} for {patient_id}")
    fu.status = "completed"
    fu.completed_by_scan = f"result:{note or fu.action}"
    for t in case.timers:
        if t.ref == fu.id:
            t.done = True
    case.log("results", f"{fu.id} '{fu.action}' completed by result {note!r}", rt.clock.iso())
    rt.store.save(case)
    return case


def advance_clock(rt: Runtime, **delta) -> list[Case]:
    """Fast-forward the simulated clock and let the Watchdog fire every due timer."""
    rt.clock.advance(**delta)
    out = []
    for case in rt.store.list():
        case = watchdog.tick(case, rt)
        rt.store.save(case)
        out.append(case)
    return out


def _activate(case: Case, rt: Runtime) -> Case:
    already = {f.id for f in case.followups if f.appointment}
    told_serious = any(m.to == "patient" and m.tier == "serious" and "will call" in m.text for m in case.messages)
    case = scheduler.book_all(case, rt)
    new_bookings = [f for f in case.followups if f.status == "booked" and f.appointment and f.id not in already]
    # Only text the patient when something changed for them; never repeat "your doctor will call".
    if not (case.message_tier == "serious" and told_serious):
        if new_bookings or case.message_tier == "serious" or not told_serious:
            case = patient_comms.notify(case, rt)
    if case.message_tier == "serious":
        booked = sorted(new_bookings, key=lambda f: f.appointment)
        if booked:
            parts = []
            for f in booked:
                label = patient_label(f.action)
                when = datetime.fromisoformat(f.appointment).strftime("%a %b %d at %I:%M %p")
                parts.append(f"{label} on {when}")
            text = f"We've booked: {'; '.join(parts)}. Reply OK to confirm or RESCHEDULE. - your care team"
            rt.messenger.send("patient", case.patient.patient_id, text)
            case.messages.append(Message(to="patient", tier="serious", text=text, at=rt.clock.iso()))
    case.state = CaseState.ACTIVE
    case.log("supervisor", "plan active; watchdog monitoring", rt.clock.iso())
    return case


def _get(rt: Runtime, patient_id: str) -> Case:
    case = rt.store.get(patient_id)
    if case is None:
        raise KeyError(f"no case for {patient_id}")
    return case
