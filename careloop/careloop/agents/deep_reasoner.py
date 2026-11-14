"""Agent 6 (deep mode) - Diagnostic Investigator, built with LangChain Deep Agents.

Enable with  llm.reasoner_backend: deepagent  (needs a real LLM key).

The investigator plans its own work (built-in write_todos), pulls evidence through tools
that read ONLY the shared case record, consults two subagents, writes a workup note to
its virtual filesystem, and must finish by calling submit_differential. That tool
validates the output and rejects unknown finding IDs so the agent can fix them; the
code Verifier still runs afterwards in the main graph. Intervals and rules still come
only from knowledge.py.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from ..knowledge import is_high_risk_lung, lung_nodule_rule, urgency_for
from ..runtime import Runtime
from ..schema import Case, CaseState, DxCandidate
from .reasoner import timeline

SYSTEM = """You are the Diagnostic Investigator in CareLoop, a decision-support system. A physician
reviews everything you produce; you never diagnose on your own.

Your job: connect a patient's imaging findings (across ALL their scans) and history into
3-5 ranked POSSIBLE diagnoses with cited evidence.

Work like this:
1. Use write_todos to plan your investigation before anything else.
2. Call get_timeline, then inspect important findings with get_finding and compare_with_prior.
3. Use get_patient_history for risk factors.
4. Ask the `radiology-critic` subagent (via the task tool) to challenge your draft ranking,
   especially whether findings could be UNRELATED or have a benign explanation.
5. Ask the `guideline-checker` subagent to check your next tests against the rules.
6. Write a short workup note to /workup.md (what you checked, what you concluded, open questions).
7. Finish by calling submit_differential exactly once. If it returns an error, fix and resubmit.

Hard rules:
- Every candidate cites finding IDs that exist (e.g. F-002). Never invent IDs.
- Include at least one benign/alternative explanation, and consider that findings may be unrelated.
- Do not rely on findings with confidence below {floor} unless you label them weak.
- If the evidence supports nothing, submit with insufficient_info=true and an empty list.
- The same lesion across scans ('same lesion as') is one observation: cite the latest ID, note the change in caveats.
- Data missing from the record goes in caveats; it is not a reason to keep investigating forever."""

CRITIC_PROMPT = """You are a skeptical attending radiologist. You receive a draft differential.
Use the tools to check the cited findings. Reply with at most 3 BLOCKING problems (evidence that doesn't
support a claim, the same lesion double-counted, ranking that contradicts the evidence, an obvious
explanation missing) and separately at most 3 non-blocking notes (data not in the record, nice-to-haves).
Be brief. If there are no blocking problems, say so clearly."""

GUIDELINE_PROMPT = """You check proposed next tests against CareLoop's coded rules. Use lookup_guideline
for any nodule or incidental finding and report what interval the rules give. Never invent intervals."""


class SubmittedDx(BaseModel):
    rank: int
    dx: str
    likelihood: str = Field("moderate", description="high | moderate | low")
    evidence_for: list[str] = Field(description="Finding IDs ONLY, e.g. ['F-002']. No prose.")
    evidence_against: list[str] = Field(default_factory=list, description="Finding IDs ONLY. No prose.")
    caveats: list[str] = Field(default_factory=list, description="Reasons, missing data, uncertainties (prose here).")
    rationale: str = ""
    next_tests: list[str] = Field(default_factory=list)


def build_tools(case: Case, rt: Runtime, sink: dict) -> dict[str, Any]:
    """Tools are closures over ONE case: the agent can read the record but only write via submit."""

    def _trace(msg: str) -> None:
        sink.setdefault("trace", []).append(msg)

    @tool
    def get_timeline() -> str:
        """Return the patient's findings across all scans in date order, plus history, labs and meds."""
        _trace("get_timeline")
        return timeline(case)

    @tool
    def get_finding(finding_id: str) -> str:
        """Return full details for one finding ID (e.g. 'F-002'), including report quote and sources."""
        _trace(f"get_finding({finding_id})")
        f = case.finding(finding_id)
        return f.model_dump_json(exclude_none=True) if f else f"ERROR: no finding {finding_id}. Valid IDs: " \
            + ", ".join(x.finding_id for x in case.findings)

    @tool
    def get_patient_history() -> str:
        """Return demographics, history, labs, medications and lung-cancer risk status."""
        _trace("get_patient_history")
        p = case.patient
        return json.dumps({"age": p.age, "sex": p.sex, "history": p.history, "labs": p.labs, "meds": p.meds,
                           "high_risk_lung": is_high_risk_lung(p.history, p.age)})

    @tool
    def compare_with_prior(finding_id: str) -> str:
        """For a finding, report whether the same lesion was seen on an earlier scan and how it changed."""
        _trace(f"compare_with_prior({finding_id})")
        f = case.finding(finding_id)
        if not f:
            return f"ERROR: no finding {finding_id}"
        if not f.linked_to:
            same_kind = [x.finding_id for x in case.findings if x.finding == f.finding and x.finding_id != f.finding_id]
            return f"No linked prior for {finding_id}. Other findings of the same type: {same_kind or 'none'}."
        p = case.finding(f.linked_to)
        return (f"{finding_id} is the same lesion as {p.finding_id} ({p.date}, {p.size_mm} mm) -> "
                f"now {f.size_mm} mm on {f.date}; growth {f.growth_mm} mm.")

    @tool
    def lookup_guideline(finding: str, size_mm: Optional[float] = None) -> str:
        """Look up CareLoop's coded follow-up rule and urgency level for a finding type and size."""
        _trace(f"lookup_guideline({finding}, {size_mm})")
        lvl = urgency_for(finding, size_mm, 0.9, True)
        out = {"urgency_level": lvl}
        if finding == "lung nodule" and size_mm:
            r = lung_nodule_rule(size_mm, is_high_risk_lung(case.patient.history, case.patient.age),
                                 rt.clock.now().date().isoformat())
            out["rule"] = {"action": r[0], "due": r[1], "source": r[2]} if r else "no routine follow-up"
        return json.dumps(out)

    @tool
    def submit_differential(differential: list[SubmittedDx], insufficient_info: bool = False) -> str:
        """Submit the final ranked differential (3-5 candidates). Call once, at the end."""
        valid = {f.finding_id for f in case.findings}
        import re as _re
        for d in differential:  # prose in ID lists -> caveats, same as the single-call reasoner
            prose = [x for x in d.evidence_for + d.evidence_against if not _re.match(r"^F-\d{3,}$", x.strip())]
            d.evidence_for = [x for x in d.evidence_for if x not in prose]
            d.evidence_against = [x for x in d.evidence_against if x not in prose]
            d.caveats = d.caveats + prose
        bad = sorted({i for d in differential for i in d.evidence_for + d.evidence_against if i not in valid})
        if bad:
            _trace(f"submit_differential REJECTED unknown IDs {bad}")
            return f"ERROR: unknown finding IDs {bad}. Valid IDs: {sorted(valid)}. Fix and resubmit."
        if not insufficient_info and not differential:
            return "ERROR: empty differential. Submit candidates or set insufficient_info=true."
        try:
            sink["differential"] = [DxCandidate(
                rank=d.rank, dx=d.dx, likelihood=d.likelihood if d.likelihood in ("high", "moderate", "low") else "moderate",
                evidence_for=d.evidence_for, evidence_against=d.evidence_against, caveats=d.caveats,
                rationale=d.rationale, next_tests=d.next_tests) for d in differential]
        except ValidationError as e:
            return f"ERROR: {e}"
        sink["insufficient_info"] = insufficient_info
        _trace(f"submit_differential accepted ({len(differential)} candidates)")
        return "Accepted. You are done; reply with a one-line summary."

    return {t.name: t for t in [get_timeline, get_finding, get_patient_history, compare_with_prior,
                                lookup_guideline, submit_differential]}


def _model(rt: Runtime):
    if getattr(rt, "deep_model", None) is not None:  # injected in tests
        return rt.deep_model
    from langchain.chat_models import init_chat_model

    c = rt.config["llm"]
    kwargs = {"model_provider": c["provider"]}
    if c.get("temperature") is not None:
        kwargs["temperature"] = c["temperature"]
    return init_chat_model(c["reasoner_model"], **kwargs)


def build_agent(case: Case, rt: Runtime, sink: dict):
    from deepagents import create_deep_agent
    from langchain.agents.middleware import TodoListMiddleware

    t = build_tools(case, rt, sink)
    floor = rt.config["pipeline"]["confidence_floor"]
    subagents = [
        {"name": "radiology-critic",
         "description": "Skeptical attending radiologist. Send it your draft ranking; it checks for unrelated "
                        "findings, missing benign explanations and weak evidence.",
         "system_prompt": CRITIC_PROMPT,
         "tools": [t["get_timeline"], t["get_finding"], t["compare_with_prior"]]},
        {"name": "guideline-checker",
         "description": "Checks proposed next tests and follow-up intervals against CareLoop's coded rules.",
         "system_prompt": GUIDELINE_PROMPT,
         "tools": [t["lookup_guideline"], t["get_finding"]]},
    ]
    return create_deep_agent(
        model=_model(rt),
        tools=[t[n] for n in ("get_timeline", "get_finding", "get_patient_history", "compare_with_prior",
                              "lookup_guideline", "submit_differential")],
        system_prompt=SYSTEM.format(floor=floor),
        subagents=subagents,
        # Planning: this deepagents version doesn't ship write_todos by default, so add it explicitly.
        middleware=[TodoListMiddleware()],
        name="diagnostic-investigator",
    )


def _file_text(files: dict, path: str) -> str:
    f = (files or {}).get(path)
    if f is None:
        return ""
    if isinstance(f, str):
        return f
    content = f.get("content") if isinstance(f, dict) else getattr(f, "content", "")
    return "\n".join(content) if isinstance(content, list) else str(content or "")


def run(case: Case, rt: Runtime) -> Case:
    sink: dict = {}
    agent = build_agent(case, rt, sink)
    ask = f"Investigate patient {case.patient.patient_id}. Urgency so far: {case.urgency}."
    if case.verifier.status == "fail" and case.verifier.notes:
        ask += "\nA reviewer rejected your previous differential. Fix these:\n- " + "\n- ".join(case.verifier.notes)
    result = agent.invoke({"messages": [{"role": "user", "content": ask}]},
                          {"recursion_limit": rt.config["pipeline"].get("deep_agent_recursion_limit", 60)})

    case.agent_plan = [f"[{t.get('status', '?')}] {t.get('content', '')}" for t in result.get("todos", []) or []]
    case.workup_note = _file_text(result.get("files"), "/workup.md")
    case.agent_trace = sink.get("trace", [])
    if "differential" not in sink and not sink.get("insufficient_info"):
        # Agent never submitted: fall back to the single-call reasoner rather than leaving the case empty.
        case.log("deep_reasoner", "agent did not submit; falling back to single-call reasoner", rt.clock.iso())
        from . import reasoner

        return reasoner.run(case, rt)
    case.differential = sorted(sink.get("differential", []), key=lambda d: d.rank)[:5]
    case.insufficient_info = bool(sink.get("insufficient_info"))
    case.state = CaseState.REASONED
    top = case.differential[0].dx if case.differential else "insufficient information"
    case.log("deep_reasoner", f"investigation done ({len(case.agent_trace)} tool steps, "
                              f"{len(case.agent_plan)} todos); top: {top}", rt.clock.iso())
    return case
