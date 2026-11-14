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
