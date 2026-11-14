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
