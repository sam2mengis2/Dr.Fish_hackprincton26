"""Service layer: the few entry points the API, dashboard, demo and OpenHospital all call."""
from __future__ import annotations

from datetime import datetime

from datetime import datetime, timedelta

from .agents import clinician_comms, intake, patient_comms, perception, read_priority, reconciler, extractor, scheduler, watchdog
from .graph import build_graph
from .knowledge import patient_label
from .runtime import Runtime
from .schema import Case, CaseState, Message, Patient, Scan, Timer


def submit_scan(rt: Runtime, patient: Patient, scan: Scan) -> Case:
    """A new scan arrives: intake, then run the graph until the doctor has been texted."""
    case = intake.ingest(rt, patient, scan)
    result = build_graph(rt).invoke({"case": case}, {"recursion_limit": 50})
    case = result["case"]
    rt.store.save(case)
    return case


def import_outside_record(rt: Runtime, patient: Patient, scan: Scan) -> Case:
    """A prior scan + report from ANOTHER hospital (shared record / image exchange). It becomes history:
    its findings are extracted so new scans can be compared against it, but it starts no new workup."""
    scan.outside = True
    case = intake.ingest(rt, patient, scan)
    for step in (perception.run, extractor.run, reconciler.run):
        case = step(case, rt)
    case.pending_scan_ids = []
    if case.state == CaseState.RECONCILED:
        case.state = CaseState.INGESTED
    case.log("intake", f"outside record imported: {scan.scan_id} ({scan.date}) from {scan.site or 'another hospital'}",
             rt.clock.iso())
    rt.store.save(case)
    return case


def acquire_scan(rt: Runtime, patient: Patient, scan: Scan) -> Case:
    """Images exist but the radiologist hasn't signed the report. CareLoop ranks the scan in the reading
    queue (order reason + image-model pre-read) and starts the report-turnaround clock. Nobody outside
    radiology sees the AI pre-read; the family doctor hears when the real report is signed."""
    scan.report_status, scan.acquired_at = "pending", rt.clock.iso()
    report_text, scan.report_text = scan.report_text, ""   # never read a report that isn't signed
    case = intake.ingest(rt, patient, scan)
    case = perception.run(case, rt)
    scan = next(s for s in case.scans if s.scan_id == scan.scan_id)
    scan.ai_preread = [{"label": f.finding, "prob": f.confidence, "location": f.location} for f in case.raw_findings]
    case.raw_findings = []
    case.pending_scan_ids = [s for s in case.pending_scan_ids if s != scan.scan_id]
    read_priority.prioritize(case, scan, rt)
    hours = rt.config["timers"].get("report_due_hours", {}).get(scan.read_priority, 72)
    due = rt.clock.now() + timedelta(hours=hours)
    case.timers.append(Timer(kind="report_overdue", ref=scan.scan_id, fire_at=due.isoformat(timespec="minutes")))
    case.timers.append(Timer(kind="report_overdue_2", ref=scan.scan_id,
                             fire_at=(due + timedelta(hours=hours)).isoformat(timespec="minutes")))
    if scan.read_priority == "stat":
        reasons = "; ".join(scan.read_priority_reasons)
        text = (f"READ NOW: {patient.patient_id} {scan.modality} {scan.body_part.lower()} ({scan.site or 'this site'}) "
                f"queued STAT - {reasons}. Report due within {hours} h.")
        rt.messenger.send("radiologist", patient.patient_id, text)
        case.messages.append(Message(to="radiologist", text=text, at=rt.clock.iso()))
    case.state = CaseState.AWAITING_REPORT
    case.log("intake", f"{scan.scan_id} acquired; report pending, due {due.isoformat(timespec='minutes')}",
             rt.clock.iso())
    rt.store.save(case)
    return case


def sign_report(rt: Runtime, patient_id: str, scan_id: str, report_text: str) -> Case:
    """The radiologist signs the report for an already-acquired scan: the normal loop runs and the
    family doctor is told the report is ready, with the plan, in one text."""
    case = _get(rt, patient_id)
    scan = next((s for s in case.scans if s.scan_id == scan_id), None)
    if scan is None:
        raise KeyError(f"no scan {scan_id} for {patient_id}")
    if scan.report_status == "final":
        raise ValueError(f"report for {scan_id} already signed")
    scan.report_text, scan.report_status, scan.reported_at = report_text, "final", rt.clock.iso()
    scan.route = intake.route(scan)
    for t in case.timers:
        if t.ref == scan_id and t.kind.startswith("report_overdue"):
            t.done = True
    hrs = (rt.clock.now() - datetime.fromisoformat(scan.acquired_at)).total_seconds() / 3600 if scan.acquired_at else 0
    case.pending_scan_ids = [scan_id]
    case.raw_findings = []
    case.verifier.attempts, case.verifier.status, case.verifier.notes = 0, "pending", []
    case.state = CaseState.INGESTED
    case.log("intake", f"{scan_id} report signed after {hrs:.1f} h -> running the loop", rt.clock.iso())
    result = build_graph(rt).invoke({"case": case}, {"recursion_limit": 50})
    case = result["case"]
    rt.store.save(case)
    return case


def reading_queue(rt: Runtime) -> list[dict]:
    """Unsigned scans across all patients: stat first, then urgent, then routine; oldest first within a level."""
    order = {"stat": 0, "urgent": 1, "routine": 2}
    now = rt.clock.now()
    rows = []
    for c in rt.store.list():
        for s in c.scans:
            if s.report_status != "pending":
                continue
            waited = (now - datetime.fromisoformat(s.acquired_at)).total_seconds() / 3600 if s.acquired_at else 0
            due_h = rt.config["timers"].get("report_due_hours", {}).get(s.read_priority, 72)
            rows.append({"patient_id": c.patient.patient_id, "name": c.patient.name, "scan_id": s.scan_id,
                         "exam": f"{s.modality} {s.body_part.lower()}", "site": s.site, "indication": s.indication,
                         "priority": s.read_priority, "reasons": s.read_priority_reasons,
                         "waiting_hours": round(waited, 1), "overdue": waited > due_h, "acquired_at": s.acquired_at})
    return sorted(rows, key=lambda r: (order.get(r["priority"], 3), r["acquired_at"] or ""))


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
    for pid in [c.patient.patient_id for c in rt.store.list()]:
        case = rt.store.get(pid)  # fresh copy: a timer on one case can hand a freed slot to another
        case = watchdog.tick(case, rt)
        rt.store.save(case)
        out.append(case)
    return out


def cancel_appointment(rt: Runtime, patient_id: str, followup_id: str | None = None, reason: str = "") -> Case:
    case = _get(rt, patient_id)
    case = scheduler.cancel(case, rt, followup_id, reason)
    rt.store.save(case)
    return case


def seed_busy_scanners(rt: Runtime, demo: dict) -> list[str]:
    """Demo: per-site backlogs (Leyat: 5-6 month CT/MRI waits) + patients already holding Riverside slots."""
    from .schema import FollowUp

    busy = demo.get("busy_scanners") or {}
    rt.config.setdefault("scheduling", {})["backlog_days"] = dict(busy.get("backlog_days", {}))
    added = []
    for w in busy.get("waiting_patients", []):
        p = Patient(**w["patient"])
        if rt.store.get(p.patient_id):
            continue
        case = Case(patient=p, state=CaseState.ACTIVE)
        case.followups.append(FollowUp(id="FU-1", action=w["action"], owner=p.ordering_doctor, due=w["due"],
                                       rule="existing booking (simulated)", status="booked",
                                       appointment=w["appointment"], site=w["site"],
                                       waitlisted=w["appointment"][:10] > w["due"],
                                       booking_note="later than due date -> on cancellation list"
                                       if w["appointment"][:10] > w["due"] else ""))
        scheduler._calendar(rt, w["site"], scheduler.resource(w["action"])).take(w["appointment"])
        case.log("scheduler", f"simulated existing booking {w['appointment']} at {w['site']}", rt.clock.iso())
        rt.store.save(case)
        added.append(p.patient_id)
    return added


def waitlist(rt: Runtime) -> list[dict]:
    """Every booked scan in the network, most urgent clinical due date first, with open offers."""
    rows = []
    for c in rt.store.list():
        for f in c.followups:
            if f.status in ("booked", "missed") and f.appointment:
                rows.append({"patient_id": c.patient.patient_id, "name": c.patient.name, "followup_id": f.id,
                             "action": f.action, "due": f.due, "appointment": f.appointment, "site": f.site,
                             "late": f.waitlisted, "note": f.booking_note, "offer": f.offer})
    return sorted(rows, key=lambda r: (r["due"], r["appointment"]))


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
                parts.append(f"{label} on {when}" + (f" at {f.site}" if f.site else ""))
            late = [patient_label(f.action) for f in booked if f.waitlisted]
            wait = (f" Your {' and '.join(late)} {'is' if len(late) == 1 else 'are'} later than we'd like, so you're on "
                    "the cancellation list: we'll text you the moment an earlier slot opens." if late else "")
            text = f"We've booked: {'; '.join(parts)}.{wait} Reply OK to confirm or RESCHEDULE. - your care team"
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


def metrics(rt: Runtime) -> dict:
    """Impact numbers for the demo: report turnaround, days saved by cancellations, human vs automated steps."""
    cases = rt.store.list()
    turn = []
    for c in cases:
        for s in c.scans:
            if s.acquired_at and s.reported_at:
                h = (datetime.fromisoformat(s.reported_at) - datetime.fromisoformat(s.acquired_at)).total_seconds() / 3600
                turn.append((s.read_priority or "routine", h))
    moved = [(c.patient.patient_id, f.days_saved) for c in cases for f in c.followups if f.days_saved > 0]
    back_on_time = [p for c in cases for f in c.followups if f.days_saved > 0 and not f.waitlisted
                    for p in [c.patient.patient_id]]
    human = sum(1 for c in cases if not c.patient.demo_auto_reply for a in c.audit_log
                if (a.agent == "clinician_comms" and a.action.startswith("doctor replied"))
                or (a.agent == "patient_comms" and a.action.startswith("patient replied")))
    automated = sum(len(c.messages) for c in cases if not c.patient.demo_auto_reply) + \
        sum(1 for c in cases if not c.patient.demo_auto_reply for a in c.audit_log if a.agent == "scheduler")
    return {
        "reports_signed": len(turn),
        "avg_turnaround_h": round(sum(h for _, h in turn) / len(turn), 1) if turn else None,
        "stat_turnaround_h": round(min((h for p, h in turn if p == "stat"), default=0), 1) if any(
            p == "stat" for p, _ in turn) else None,
        "unsigned": sum(1 for c in cases for s in c.scans if s.report_status == "pending"),
        "days_saved": sum(d for _, d in moved),
        "patients_moved_up": len(moved),
        "back_on_time": len(set(back_on_time)),
        "human_actions": human,
        "automated_steps": automated,
        "deep_agent_calls": sum(c.agent_llm_calls for c in cases),
    }
