"""Leyat's pain points (Mount Sinai imaging): report delays, cross-hospital priors and scheduling."""
import json

from careloop import service
from careloop.agents import watchdog
from careloop.runtime import ROOT, CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
from careloop.schema import Patient, Scan

DATA = json.loads((ROOT / "data" / "demo_cases.json").read_text())


def _rt(tmp_path):
    cfg = load_config()
    cfg["llm"]["provider"] = "mock"; cfg["cv"]["cxr_backend"] = "mock"
    return Runtime(cfg, CaseStore(tmp_path), SimClock("2026-11-14T09:00"), ConsoleMessenger(quiet=True))


def P(pid):
    return Patient(**DATA["patients"][pid])


def S(sid):
    e = next(e for e in DATA["scans"] + DATA["outside_records"] if e["scan"]["scan_id"] == sid)
    return Scan(**e["scan"])


def test_reading_queue_puts_stat_first_and_hides_unsigned_reports(tmp_path):
    rt = _rt(tmp_path)
    service.acquire_scan(rt, P("P-004"), S("CXR-P004-1"))   # routine, arrived first
    rt.clock.advance(minutes=10)
    service.acquire_scan(rt, P("P-002"), S("CT-P002-1"))    # routine
    rt.clock.advance(minutes=10)
    c = service.acquire_scan(rt, P("P-003"), S("CXR-P003-1"))  # pneumothorax: arrived last
    q = service.reading_queue(rt)
    assert [r["patient_id"] for r in q][0] == "P-003" and q[0]["priority"] == "stat"
    assert [r["patient_id"] for r in q][1:] == ["P-004", "P-002"]          # then first-come
    assert any(m.to == "radiologist" and "READ NOW" in m.text for m in c.messages)
    # Nothing reaches the family doctor or patient before the report is signed.
    for pid in ("P-002", "P-003", "P-004"):
        case = rt.store.get(pid)
        assert case.state.value == "awaiting_report"
        assert not [m for m in case.messages if m.to in ("doctor", "patient")]
        assert not case.findings and all(s.report_text == "" for s in case.scans)


def test_overdue_report_alerts_radiology_then_tells_doctor(tmp_path):
    rt = _rt(tmp_path)
    service.acquire_scan(rt, P("P-002"), S("CT-P002-1"))    # routine: due in 72 h
    rt.clock.advance(hours=73)
    c = watchdog.tick(rt.store.get("P-002"), rt)
    assert any(m.to == "radiologist" and "overdue" in m.text.lower() for m in c.messages)
    assert not any(m.to == "doctor" for m in c.messages)
    rt.clock.advance(hours=72)
    c = watchdog.tick(c, rt)
    assert any(m.to == "doctor" and "delayed" in m.text for m in c.messages)
    assert c.state.value == "awaiting_report"               # never closed while a report is pending


def test_signed_report_runs_loop_and_tells_doctor_report_ready(tmp_path):
    rt = _rt(tmp_path)
    service.acquire_scan(rt, P("P-002"), S("CT-P002-1"))
    rt.clock.advance(hours=5)
    c = service.sign_report(rt, "P-002", "CT-P002-1", S("CT-P002-1").report_text)
    doc = [m.text for m in c.messages if m.to == "doctor"][-1]
    assert doc.startswith("CareLoop") and "Report ready" in doc
    assert c.state.value in ("awaiting_approval", "needs_human_review")
    assert not service.reading_queue(rt)
    assert not any(t.kind.startswith("report_overdue") and not t.done for t in c.timers)


def test_outside_prior_shows_growth_and_moves_followup_sooner(tmp_path):
    rt = _rt(tmp_path)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    c = rt.store.get("P-002")
    assert not [m for m in c.messages]                      # history only: nobody is texted
    c = service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    nod = next(f for f in c.findings if f.finding == "lung nodule" and f.scan_id == "CT-P002-1")
    assert nod.linked_to and nod.growth_mm == 3
    fu = next(f for f in c.followups if "chest ct" in f.action.lower())
    assert "grew" in fu.rule and fu.due == "2027-02-11"       # 3 months, not 6-12
    doc = [m.text for m in c.messages if m.to == "doctor"][-1]
    assert "Riverside Medical Center" in doc and "+3 mm" in doc


def test_booking_prefers_patient_site_and_switch_moves_it(tmp_path):
    rt = _rt(tmp_path)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    c, _ = service.doctor_reply(rt, "P-002", "APPROVE")
    ct = next(f for f in c.followups if "chest ct" in f.action.lower())
    assert ct.site == "Riverside Medical Center"
    pt = [m.text for m in c.messages if m.to == "patient"][-1]
    assert "Riverside Medical Center" in pt and "SWITCH" in pt
    c, outcome = service.patient_reply(rt, "P-002", "SWITCH")
    ct = next(f for f in c.followups if "chest ct" in f.action.lower())
    assert outcome == "SWITCHED_SITE" and ct.site == "Eastside General" and ct.status == "booked"
    assert ct.appointment[:10] <= ct.due


def test_followup_scan_at_other_hospital_closes_the_loop(tmp_path):
    rt = _rt(tmp_path)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    c, _ = service.doctor_reply(rt, "P-002", "APPROVE")
    ct = next(f for f in c.followups if "chest ct" in f.action.lower())
    rt.clock.set(ct.appointment)
    follow = S("CT-P002-2")
    follow.date = ct.appointment[:10]
    c = service.submit_scan(rt, P("P-002"), follow)
    assert c.followup(ct.id).status == "completed" and c.followup(ct.id).completed_by_scan == "CT-P002-2"


def _busy_maria(tmp_path):
    rt = _rt(tmp_path)
    service.seed_busy_scanners(rt, DATA)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    service.doctor_reply(rt, "P-002", "APPROVE")
    return rt


def test_backlog_books_earliest_hospital_and_waitlists_when_late(tmp_path):
    rt = _busy_maria(tmp_path)
    c = rt.store.get("P-002")
    ct = next(f for f in c.followups if "chest ct" in f.action.lower())
    assert ct.site == "Riverside Medical Center" and ct.waitlisted       # 120-day line beats 165, still late
    assert "cancellation list" in [m.text for m in c.messages if m.to == "patient"][-1]


def test_skipped_offer_goes_to_next_patient(tmp_path):
    rt = _busy_maria(tmp_path)
    service.cancel_appointment(rt, "P-101")
    service.patient_reply(rt, "P-002", "SKIP")
    rows = {r["patient_id"]: r for r in service.waitlist(rt)}
    assert rows["P-002"]["appointment"].startswith("2027-03-15")       # Maria kept her slot
    assert rows["P-103"]["appointment"].startswith("2026-12-03")       # next most urgent got it (auto-accepts)


def test_unanswered_offer_expires_and_moves_on(tmp_path):
    rt = _busy_maria(tmp_path)
    service.cancel_appointment(rt, "P-101")
    service.advance_clock(rt, hours=5)                                    # held 4 h
    rows = {r["patient_id"]: r for r in service.waitlist(rt)}
    assert not rows["P-002"]["offer"] and rows["P-103"]["appointment"].startswith("2026-12-03")
    assert any("passed to the next patient" in m.text for m in rt.store.get("P-002").messages)


def test_scanner_backlog_never_delays_clinic_visits_or_takes_their_offers(tmp_path):
    from careloop.agents import planner, scheduler

    rt = _busy_maria(tmp_path)
    c = rt.store.get("P-002")
    # the real model's plan for a growing nodule also includes a specialist visit, due soon
    planner._add(c, "Pulmonology referral", "2026-12-01", "Top candidate next test; urgent", [])
    scheduler.book_all(c, rt)
    rt.store.save(c)
    visit = next(f for f in c.followups if "pulmonology" in f.action.lower())
    assert scheduler.resource(visit.action) == "clinic" and visit.appointment < "2026-12-02"   # no 4-month wait
    service.cancel_appointment(rt, "P-101")          # frees a CT slot
    c = rt.store.get("P-002")
    offered = [f for f in c.followups if f.offer]
    assert [scheduler.resource(f.action) for f in offered] == ["CT"]  # the CT item, not the earlier-due visit


def test_cross_region_link_always_shows_unrelated_option(tmp_path):
    from careloop.agents import verifier
    from careloop.schema import DxCandidate

    rt = _rt(tmp_path)
    service.submit_scan(rt, P("P-001"), S("CXR-P001-1"))
    service.doctor_reply(rt, "P-001", "APPROVE")
    c = service.submit_scan(rt, P("P-001"), S("MRI-P001-1"))
    lung = next(f.finding_id for f in c.findings if "lung" in f.finding)
    brain = next(f.finding_id for f in c.findings if "brain" in f.finding)
    c.differential = [DxCandidate(rank=1, dx="Metastatic lung cancer to brain", likelihood="high",
                                  evidence_for=[lung, brain])]
    verifier.ensure_unrelated_option(c, rt)
    assert any("unrelated" in d.dx.lower() for d in c.differential)
    n = len(c.differential)
    verifier.ensure_unrelated_option(c, rt)
    assert len(c.differential) == n                  # added once, not repeatedly


def test_verifier_stops_repeat_complaint_loops_but_never_softens_code_checks(tmp_path, monkeypatch):
    from careloop.agents import verifier

    rt = _rt(tmp_path)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    c = service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    c.verifier.attempts, c.verifier.history, c.verifier.suggestions = 0, [], []
    said = iter(["Benign candidate has no 'against' evidence although growth argues against a benign nodule.",
                 "The benign nodule candidate lacks 'against' evidence even though the growth argues against benign."])
    monkeypatch.setattr(verifier, "_mock_critic", lambda case: [next(said)])
    c = verifier.run(c, rt)
    assert c.verifier.status == "fail"                       # first complaint: one redo
    c = verifier.run(c, rt)
    assert c.verifier.status == "pass" and "growth" in c.verifier.suggestions[0]   # repeat -> reviewer note
    # an invented finding ID is a code check: it always blocks, however often it repeats
    c.differential[0].evidence_for.append("F-999")
    c = verifier.run(c, rt)
    assert c.verifier.status == "fail"


def test_deep_agent_gets_one_redo_and_guideline_tool_sees_growth(tmp_path):
    import json as _json

    from careloop.agents import deep_reasoner, verifier

    rt = _rt(tmp_path)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    c = service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    tools = deep_reasoner.build_tools(c, rt, {})
    out = _json.loads(tools["lookup_guideline"].invoke({"finding": "lung nodule", "size_mm": 7}))
    assert out["growth_mm"] == 3 and out["rule"]["due"] == "2027-02-11"   # same rule the planner books
    c.agent_trace = ["get_timeline"]
    c.verifier.attempts, c.verifier.history = 0, []
    c.differential[0].evidence_for.append("F-999")
    c = verifier.run(c, rt)
    assert c.state.value == "verify_failed"                  # one redo allowed
    c = verifier.run(c, rt)
    assert c.state.value == "needs_human_review"             # not a third multi-minute investigation


def test_ai_urgency_keeps_guideline_date_and_never_books_invasive_procedures(tmp_path):
    from careloop.agents import planner, scheduler
    from careloop.schema import CaseState, DxCandidate

    rt = _rt(tmp_path)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    c = service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    nod = next(f for f in c.findings if f.scan_id == "CT-P002-1" and f.finding == "lung nodule")
    c.urgency, c.state = "urgent", CaseState.VERIFIED
    c.differential = [DxCandidate(rank=1, dx="Primary lung malignancy", likelihood="moderate",
                                  evidence_for=[nod.finding_id],
                                  next_tests=["Short-interval chest CT", "PET-CT", "CT-guided biopsy"])]
    c = planner.run(c, rt)
    ct = next(f for f in c.followups if "chest ct" in f.action.lower())
    assert ct.due == "2027-02-11"                                   # guideline date, not the urgent window
    pet = next(f for f in c.followups if "pet" in f.action.lower())
    assert pet.due == "2026-11-28"                                  # extra AI test: within 2 weeks
    bx = next(f for f in c.followups if "biops" in f.action.lower())
    assert bx.action.startswith("Discuss") and "clinician decision, never auto-booked" in bx.rule
    scheduler.book_all(c, rt)
    assert bx.appointment is None and ct.appointment and pet.appointment


def test_followup_scan_text_says_loop_closed(tmp_path):
    rt = _rt(tmp_path)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    c, _ = service.doctor_reply(rt, "P-002", "APPROVE")
    ct = next(f for f in c.followups if "chest ct" in f.action.lower())
    rt.clock.set(ct.appointment)
    follow = S("CT-P002-2")
    follow.date = ct.appointment[:10]
    c = service.submit_scan(rt, P("P-002"), follow)
    doc = [m.text for m in c.messages if m.to == "doctor"][-1]
    assert "Follow-up done:" in doc and "stable at 7 mm" in doc


def test_serious_tier_booking_text_has_hospital_and_cancellation_list(tmp_path, monkeypatch):
    from careloop.agents import planner

    rt = _rt(tmp_path)
    service.seed_busy_scanners(rt, DATA)
    service.import_outside_record(rt, P("P-002"), S("CT-P002-0"))
    real_run = planner.run

    def serious(case, rt):
        case = real_run(case, rt)
        case.message_tier = "serious"   # e.g. the real model ranks malignancy first
        return case
    monkeypatch.setattr(planner, "run", serious)
    import careloop.graph as g
    monkeypatch.setattr(g, "planner", planner, raising=False)
    service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    c, _ = service.doctor_reply(rt, "P-002", "APPROVE")
    booked = [m.text for m in c.messages if m.to == "patient" and m.text.startswith("We've booked")]
    assert c.message_tier == "serious"
    assert booked and "Riverside Medical Center" in booked[-1] and "cancellation list" in booked[-1]


def test_demo_time_skip_only_moves_this_patient(tmp_path):
    from fastapi.testclient import TestClient

    import api
    from careloop.runtime import CaseStore

    api.rt.store = CaseStore(tmp_path)
    c = TestClient(api.app)
    c.post("/reset")
    c.post("/demo/busy")
    for pid in ("P-004", "P-002", "P-002"):
        c.post(f"/demo/load/{pid}")
    c.post("/reply/doctor", json={"patient_id": "P-002", "text": "APPROVE"})
    c.post("/demo/load/P-002")                         # follow-up CT: time skip to the appointment
    grace = c.get("/cases/P-004").json()
    assert not [m for m in grace["messages"] if "overdue" in m["text"].lower()]   # untouched patient stays quiet
    maria = c.get("/cases/P-002").json()
    assert any(f["status"] == "completed" and f["completed_by_scan"] == "CT-P002-2" for f in maria["followups"])
