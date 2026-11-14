"""Run with: pytest -q   (mock mode, ~1 second)"""
from careloop.runtime import load_config
from openhospital.env import OpenHospitalEnv
from openhospital.tasks import TASKS


def _run(task_id, **overrides):
    cfg = load_config()
    cfg["llm"]["provider"] = "mock"
    for k, v in overrides.items():
        sec, key = k.split("__")
        cfg[sec][key] = v
    env = OpenHospitalEnv(cfg)
    env.reset(task_id)
    r = env.run()
    env.close()
    return r


def test_all_tasks_pass_in_mock_mode():
    for tid in TASKS:
        r = _run(tid)
        assert r.error is None, (tid, r.error)
        assert r.score == 1.0, (tid, r.checks)


def test_verifier_catches_invented_evidence():
    with_v = _run("T8_invented_evidence")
    without_v = _run("T8_invented_evidence", pipeline__verifier_enabled=False)
    assert with_v.checks["final_differential_cites_only_real_ids"]
    assert not without_v.checks["final_differential_cites_only_real_ids"]


def test_verbatim_guard_drops_hallucinated_quotes():
    from careloop.agents.extractor import _mock_extract

    report = "There is a 7 mm nodule in the right lower lobe. No pneumothorax."
    found = _mock_extract(report)
    assert [f.finding for f in found] == ["lung nodule"]          # negated pneumothorax ignored
    assert all(f.report_quote in report for f in found)


def test_fleischner_simplified_rules():
    from careloop.knowledge import lung_nodule_rule

    assert lung_nodule_rule(4, high_risk=False, scan_date="2026-01-01") is None
    assert lung_nodule_rule(7, high_risk=False, scan_date="2026-01-01")[1] == "2026-07-02"
    assert "3 months" in lung_nodule_rule(12, high_risk=True, scan_date="2026-01-01")[2]


def test_no_patient_text_ever_names_a_serious_step():
    """Regression: a missed-appointment text once said 'Oncology referral' to a serious-tier patient."""
    from careloop.knowledge import SERIOUS_WORDS, patient_label

    assert patient_label("Oncology referral") == "specialist appointment"
    assert patient_label("Tissue biopsy of lung mass") == "procedure"
    assert patient_label("PET-CT") == "PET-CT"
    r = _run("T7_no_diagnosis_leak")
    assert r.checks["no_serious_words_to_patient"]


def test_prose_in_id_lists_is_moved_to_caveats():
    """Regression (real LLM, 2026-11-14): the model wrote reasons into evidence_against."""
    from careloop.schema import DxCandidate, tidy_differential

    d = DxCandidate(rank=1, dx="Benign nodule", evidence_for=["F-002", "small and smooth"],
                    evidence_against=["No prior imaging for comparison", "F-001"])
    tidy_differential([d])
    assert d.evidence_for == ["F-002"] and d.evidence_against == ["F-001"]
    assert "No prior imaging for comparison" in d.caveats and "small and smooth" in d.caveats


def test_guideline_followup_survives_failed_reasoning():
    """Regression: when the Verifier gives up, the Fleischner follow-up must still be created."""
    r_cfg = {"pipeline__inject_reasoner_fault": True, "pipeline__max_reasoner_retries": 0}
    from careloop import service
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from openhospital.tasks import P, S
    import tempfile

    cfg = load_config()
    cfg["llm"]["provider"] = "mock"
    for k, v in r_cfg.items():
        a, b = k.split("__"); cfg[a][b] = v
    rt = Runtime(cfg, CaseStore(tempfile.mkdtemp()), SimClock("2026-11-14T09:00"), ConsoleMessenger(quiet=True))
    c = service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    assert c.state.value == "needs_human_review"
    assert any("chest ct" in f.action.lower() and "Fleischner" in f.rule for f in c.followups)
    c, _ = service.doctor_reply(rt, "P-002", "APPROVE")
    pt = [m.text for m in c.messages if m.to == "patient"]
    assert pt and not any("do not need" in t.lower() for t in pt)


def test_critic_suggestions_do_not_fail_review():
    """Regression (real LLM, 2026-11-14): the critic failed every case over data not in the record."""
    from careloop.agents import verifier
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from careloop.schema import Case, DxCandidate, Finding, Patient
    import tempfile

    class StubLLM:
        is_mock, calls = False, 0

        def structured(self, role, system, user, schema):
            return schema(blocking=[], suggestions=["No smoking history in the record"])

    cfg = load_config()
    rt = Runtime(cfg, CaseStore(tempfile.mkdtemp()), SimClock(), ConsoleMessenger(quiet=True), llm=StubLLM())
    c = Case(patient=Patient(patient_id="X", name="A B", age=50),
             findings=[Finding(finding_id="F-001", finding="lung nodule", confidence=0.9, scan_id="S1")],
             differential=[DxCandidate(rank=1, dx="Benign nodule", evidence_for=["F-001"], next_tests=["CT"])])
    c = verifier.run(c, rt)
    assert c.verifier.status == "pass"
    assert c.verifier.suggestions == ["No smoking history in the record"]


def test_ids_with_notes_are_kept():
    """Regression: 'F-003 (stable)' lost its ID and the Verifier failed for 'no evidence'."""
    from careloop.schema import DxCandidate, tidy_differential

    d = DxCandidate(rank=1, dx="x", evidence_for=["F-003 (stable over 6 months)", "F-001, F-002"])
    tidy_differential([d])
    assert d.evidence_for == ["F-003", "F-001", "F-002"]
    assert "F-003 (stable over 6 months)" in d.caveats


def test_text_blocks_extracted_from_thinking_output():
    """Regression: a patient text was the raw [{'signature':..., 'thinking':...}, {'text':...}] list."""
    from careloop.llm import content_text

    blocks = [{"type": "thinking", "thinking": "", "signature": "abc"}, {"type": "text", "text": "Hi Maria"}]
    assert content_text(blocks) == "Hi Maria"


def test_conditional_tests_not_booked_and_dates_respected():
    from careloop.agents import planner, scheduler
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from careloop.schema import Case, DxCandidate, FollowUp, Patient
    import tempfile

    rt = Runtime(load_config(), CaseStore(tempfile.mkdtemp()), SimClock("2026-11-14T09:00"), ConsoleMessenger(quiet=True))
    c = Case(patient=Patient(patient_id="X", name="A B", age=50), urgency="routine",
             followups=[FollowUp(id="FU-1", action="Follow-up chest CT", owner="Dr", due="2027-05-14")],
             differential=[DxCandidate(rank=1, dx="stone", evidence_for=[], next_tests=[
                 "Urinalysis", "Serum creatinine", "Renal ultrasound if the stone does not pass",
                 "Review CT for hydronephrosis", "Urology consultation if obstruction"])])
    c = planner.run(c, rt)
    actions = [f.action for f in c.followups]
    assert "Urinalysis" in actions and not any(" if " in a or a.startswith("Review CT") for a in actions)
    c = scheduler.book_all(c, rt)
    for f in c.followups:
        if f.appointment:
            assert f.appointment[:10] <= f.due, (f.action, f.appointment, f.due)


def test_lab_results_close_followups_and_doctor_sees_incidentals():
    from careloop import service
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from openhospital.tasks import P, S
    import tempfile

    rt = Runtime(load_config(), CaseStore(tempfile.mkdtemp()), SimClock("2026-11-14T09:00"), ConsoleMessenger(quiet=True))
    c = service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    doc = [m.text for m in c.messages if m.to == "doctor"][-1]
    assert "lung nodule" in doc and "kidney stone" in doc        # incidental nodule is visible
    fu = c.followups[0]
    c = service.record_result(rt, "P-002", fu.id, "normal")
    assert c.followup(fu.id).status == "completed"


def test_bilateral_report_location_merges_with_model_finding():
    """Regression: 'left frontal and right parietal' (report) vs 'left frontal' (model) didn't merge."""
    from careloop.agents import reconciler
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from careloop.schema import Case, Finding, Patient
    import tempfile

    rt = Runtime(load_config(), CaseStore(tempfile.mkdtemp()), SimClock(), ConsoleMessenger(quiet=True))
    c = Case(patient=Patient(patient_id="X", name="A B", age=50), raw_findings=[
        Finding(finding="ring-enhancing brain lesion", location="left frontal lobe and right parietal lobe",
                confidence=0.9, sources=["report_extraction"], scan_id="M1"),
        Finding(finding="ring-enhancing brain lesion", location="left frontal lobe", confidence=0.91,
                sources=["brain_mri_model"], scan_id="M1")])
    c = reconciler.run(c, rt)
    assert len(c.findings) == 1 and set(c.findings[0].sources) == {"brain_mri_model", "report_extraction"}
    assert not c.discrepancies


def test_needs_review_cases_are_chased():
    """Regression: a needs-human-review case got no reminder or escalation."""
    from careloop import service
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from openhospital.tasks import P, S
    import tempfile

    cfg = load_config()
    cfg["pipeline"]["inject_reasoner_fault"] = True
    cfg["pipeline"]["max_reasoner_retries"] = 0
    rt = Runtime(cfg, CaseStore(tempfile.mkdtemp()), SimClock("2026-11-14T09:00"), ConsoleMessenger(quiet=True))
    c = service.submit_scan(rt, P("P-001"), S("CXR-P001-1"))
    assert c.state.value == "needs_human_review"
    assert "FYI" not in [m.text for m in c.messages if m.to == "doctor"][-1]
    service.advance_clock(rt, minutes=31)
    c = rt.store.get("P-001")
    assert any(m.to == "covering_doctor" for m in c.messages)


def test_differently_worded_duplicate_tests_are_not_rebooked():
    """Regression (real LLM): 3 biopsies and 3 chest CTs got booked across successive scans."""
    from careloop.knowledge import test_type

    biopsies = ["Bronchoscopy with biopsy", "EBUS-TBNA or CT-guided biopsy of hilar mass",
                "CT-guided or bronchoscopic biopsy of the right hilar mass with molecular testing"]
    cts = ["CT chest with contrast", "CT chest/abdomen with contrast", "Contrast-enhanced CT chest/abdomen/pelvis"]
    assert {test_type(b) for b in biopsies} == {"biopsy"}
    assert {test_type(c) for c in cts} == {"ct-chest"}
    assert test_type("PET-CT") == "pet-ct" and test_type("Chest CT") == "ct-chest"


def test_urgent_need_moves_pending_routine_test_earlier():
    from careloop.agents.planner import _add
    from careloop.schema import Case, FollowUp, Patient

    c = Case(patient=Patient(patient_id="X", name="A B", age=60),
             followups=[FollowUp(id="FU-1", action="Follow-up chest CT", owner="Dr", due="2027-05-14",
                                 appointment="2027-05-07T09:00", status="booked")])
    _add(c, "CT chest with contrast", "2026-11-21", "urgent", [])
    assert len(c.followups) == 1
    assert c.followups[0].due == "2026-11-21" and c.followups[0].appointment is None


def test_fault_injection_applies_to_llm_backend_too():
    """Regression: T8 only planted the fake ID in mock mode, so real-LLM T8 passes were hollow."""
    from careloop.agents import reasoner
    from careloop.agents.reasoner import ReasonerOut
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from careloop.schema import Case, DxCandidate, Finding, Patient
    import tempfile

    class StubLLM:
        is_mock, calls = False, 0

        def structured(self, role, system, user, schema):
            return ReasonerOut(differential=[DxCandidate(rank=1, dx="x", evidence_for=["F-001"], next_tests=["CT"])])

    cfg = load_config()
    cfg["pipeline"]["inject_reasoner_fault"] = True
    rt = Runtime(cfg, CaseStore(tempfile.mkdtemp()), SimClock(), ConsoleMessenger(quiet=True), llm=StubLLM())
    c = Case(patient=Patient(patient_id="X", name="A B", age=50),
             findings=[Finding(finding_id="F-001", finding="lung nodule", scan_id="S")])
    c = reasoner.run(c, rt)
    assert "F-999" in c.differential[0].evidence_for


def test_patient_label_names_tests_plainly():
    from careloop.knowledge import patient_label
    assert patient_label("MRI brain with and without gadolinium (dedicated metastasis protocol)") == \
        "MRI brain with and without gadolinium"
    assert patient_label("Bronchoscopy with biopsy (EBUS-TBNA)") == "procedure"


def test_deep_agent_commentary_and_clinician_tasks():
    """Regression (Deep Agent, real model): a radiologist task was booked and texted to the patient."""
    from careloop.knowledge import clean_action, is_clinician_task

    assert clean_action("Tissue biopsy (EBUS or CT-guided) - clinician decision") == "Tissue biopsy (EBUS or CT-guided)"
    assert clean_action("Multidisciplinary/pulmonology referral (timeframe is clinical judgment, not a coded rule)") \
        == "Multidisciplinary/pulmonology referral"
    assert is_clinician_task("Confirm whether the completed MRI included DWI/ADC and SWI")
    assert is_clinician_task("Neurosurgery/radiation oncology input via the MDT")
    assert not is_clinician_task("PET-CT") and not is_clinician_task("Tissue biopsy (EBUS or CT-guided)")


def test_clinician_tasks_not_booked_or_texted():
    from careloop.agents import planner, scheduler
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from careloop.schema import Case, DxCandidate, Patient
    import tempfile

    rt = Runtime(load_config(), CaseStore(tempfile.mkdtemp()), SimClock("2026-11-14T09:00"), ConsoleMessenger(quiet=True))
    c = Case(patient=Patient(patient_id="X", name="A B", age=60), urgency="urgent",
             differential=[DxCandidate(rank=1, dx="x", evidence_for=[], next_tests=[
                 "PET-CT", "Confirm whether the completed MRI included DWI/ADC and SWI"])])
    c = scheduler.book_all(planner.run(c, rt), rt)
    task = next(f for f in c.followups if f.action.startswith("Confirm"))
    assert task.appointment is None and "clinician task" in task.rule
    assert next(f for f in c.followups if f.action == "PET-CT").appointment


def test_growth_not_computed_across_modalities():
    """Regression (found by the Deep Agent): 32 mm on X-ray vs 34 mm on CT is not 'growth'."""
    from careloop.agents import reconciler
    from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
    from careloop.schema import Case, Finding, Patient, Scan
    import tempfile

    rt = Runtime(load_config(), CaseStore(tempfile.mkdtemp()), SimClock(), ConsoleMessenger(quiet=True))
    c = Case(patient=Patient(patient_id="X", name="A B", age=60),
             scans=[Scan(scan_id="CXR", modality="CR", body_part="CHEST", date="2026-10-30"),
                    Scan(scan_id="CT", modality="CT", body_part="CHEST", date="2026-11-16")],
             findings=[Finding(finding_id="F-001", finding="lung mass", location="right hilum", size_mm=32,
                               scan_id="CXR", sources=["report_extraction"])],
             raw_findings=[Finding(finding="lung mass", location="right hilum", size_mm=34, scan_id="CT",
                                   sources=["report_extraction"])])
    c = reconciler.run(c, rt)
    new = c.findings[-1]
    assert new.linked_to == "F-001" and new.growth_mm is None


def test_specialist_types_and_labels():
    from careloop.knowledge import patient_label, test_type
    assert test_type("Radiation oncology referral") != test_type("Oncology referral")
    assert patient_label("Neurosurgery referral") == "specialist appointment"
