"""Deep Agent investigator, tested offline with a scripted fake chat model.

Checks the real wiring: planning (write_todos), case tools, a subagent call (task),
writing /workup.md, submit_differential rejecting an invented ID, then accepting a fix.
"""
import itertools

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from careloop import service
from careloop.agents import deep_reasoner
from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
from careloop.schema import Patient, Scan
from openhospital.tasks import DEMO


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # the fake ignores tool schemas; the script supplies calls
        return self


_ids = itertools.count()


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{next(_ids)}", "type": "tool_call"}])


def _case_after_two_scans(tmp_path):
    cfg = load_config()
    cfg["llm"]["provider"] = "mock"
    rt = Runtime(cfg, CaseStore(tmp_path), SimClock("2026-11-14T09:00"), ConsoleMessenger(quiet=True))
    p = Patient(**DEMO["patients"]["P-001"])
    for sid in ("CXR-P001-1", "MRI-P001-1"):
        s = Scan(**next(x["scan"] for x in DEMO["scans"] if x["scan"]["scan_id"] == sid))
        service.submit_scan(rt, p, s)
    return rt, rt.store.get("P-001")


def test_deep_investigator_end_to_end(tmp_path):
    rt, case = _case_after_two_scans(tmp_path)
    lung = next(f.finding_id for f in case.findings if f.finding == "lung mass")
    brain = next(f.finding_id for f in case.findings if "brain" in f.finding)
    good = [
        {"rank": 1, "dx": "Primary lung cancer with brain metastases", "likelihood": "high",
         "evidence_for": [lung, brain], "next_tests": ["PET-CT", "Oncology referral"]},
        {"rank": 2, "dx": "Brain abscesses from a lung infection", "likelihood": "low",
         "evidence_for": [brain, lung], "next_tests": ["Blood cultures"]},
        {"rank": 3, "dx": "Two unrelated processes", "likelihood": "low",
         "evidence_for": [lung, brain], "next_tests": ["Neuro-oncology review"]},
    ]
    bad = [dict(good[0], evidence_for=[lung, "F-999"])]
    script = [
        call("write_todos", todos=[{"content": "Pull timeline", "status": "in_progress"},
                                   {"content": "Compare lesions with priors", "status": "pending"},
                                   {"content": "Get critic review", "status": "pending"},
                                   {"content": "Submit differential", "status": "pending"}]),
        call("get_timeline"),
        call("compare_with_prior", finding_id=brain),
        call("task", description="Challenge: lung mass + 3 ring-enhancing brain lesions -> metastases?",
             subagent_type="radiology-critic"),
        AIMessage(content="Plausible. Make sure abscess and unrelated processes are listed."),  # subagent reply
        call("write_file", file_path="/workup.md",
             content="# Workup\nLung mass then 3 ring-enhancing lesions. Critic: list abscess + unrelated."),
        call("submit_differential", differential=bad),      # invented ID -> must be rejected
        call("submit_differential", differential=good),
        AIMessage(content="Submitted 3 candidates."),
    ]
    rt.deep_model = ScriptedModel(messages=iter(script))
    case = deep_reasoner.run(case, rt)

    assert [d.dx for d in case.differential][0].startswith("Primary lung cancer")
    assert all(i in {f.finding_id for f in case.findings} for d in case.differential for i in d.evidence_for)
    assert any("REJECTED" in t for t in case.agent_trace)            # the bad submission was caught
    assert "get_timeline" in case.agent_trace
    assert len(case.agent_plan) == 4
    assert "Workup" in case.workup_note
    assert case.state.value == "reasoned"


def test_falls_back_when_agent_never_submits(tmp_path):
    rt, case = _case_after_two_scans(tmp_path)
    rt.deep_model = ScriptedModel(messages=iter([call("get_timeline"), AIMessage(content="I give up.")]))
    case = deep_reasoner.run(case, rt)
    assert case.differential  # single-call reasoner (mock) filled it in
    assert any("falling back" in e.action for e in case.audit_log)
