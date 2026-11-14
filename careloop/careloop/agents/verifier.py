"""Agent 7 - Verifier (critic): code checks first, then an LLM critic.

Pass -> Care Planner. Fail -> back to Reasoner with notes (max retries from config),
then NEEDS_HUMAN_REVIEW.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import maybe
from ..runtime import Runtime
from ..schema import Case, CaseState

CRITIC = """You are a skeptical attending radiologist reviewing an AI's differential diagnosis.
Check: (1) could the cited findings be unrelated? (2) is a common benign explanation missing?
(3) is the ranking consistent with the evidence and confidence values?
Return ok=false with specific, actionable notes if anything is wrong."""


class Critique(BaseModel):
    ok: bool
    notes: list[str] = Field(default_factory=list)


def code_checks(case: Case, floor: float) -> list[str]:
    notes = []
    ids = {f.finding_id for f in case.findings}
    if case.insufficient_info:
        return notes
    if not case.differential:
        return ["no differential produced"]
    for d in case.differential:
        missing = [i for i in d.evidence_for + d.evidence_against if i not in ids]
        if missing:
            notes.append(f"'{d.dx}' cites finding IDs that do not exist: {missing}")
        if d.evidence_for and all((case.finding(i).confidence if case.finding(i) else 0) < floor
                                  for i in d.evidence_for):
            notes.append(f"'{d.dx}' rests only on findings below confidence {floor}")
    if not case.differential[0].next_tests:
        notes.append("top candidate has no next tests")
    return notes


def _mock_critic(case: Case) -> list[str]:
    cited = [case.finding(i) for d in case.differential[:1] for i in d.evidence_for if case.finding(i)]
    # The same lesion tracked over time (linked_to) is one thing, not two findings to connect.
    linked = {f.finding_id for f in cited if f.linked_to} | {f.linked_to for f in cited if f.linked_to}
    scans_cited = {f.scan_id for f in cited if f.finding_id not in linked}
    if linked:
        scans_cited |= {next(f.scan_id for f in cited if f.finding_id in linked)}
    mentions_unrelated = any("unrelated" in d.dx.lower() for d in case.differential)
    if len(scans_cited) > 1 and not mentions_unrelated:
        return ["top candidate links findings from different scans; add an 'unrelated findings' alternative"]
    return []


def run(case: Case, rt: Runtime) -> Case:
    floor = rt.config["pipeline"]["confidence_floor"]
    case.verifier.attempts += 1
    notes = code_checks(case, floor)
    if not notes and not case.insufficient_info:
        if maybe(rt.llm):
            from .reasoner import timeline

            diff = "\n".join(f"{d.rank}. {d.dx} [{d.likelihood}] for={d.evidence_for} against={d.evidence_against}"
                             for d in case.differential)
            c = rt.llm.structured("verifier", CRITIC, f"{timeline(case)}\n\nDifferential:\n{diff}", Critique)
            notes = [] if c.ok else c.notes
        else:
            notes = _mock_critic(case)
    case.verifier.notes = notes
    case.verifier.status = "fail" if notes else "pass"
    max_tries = rt.config["pipeline"]["max_reasoner_retries"] + 1
    if notes and case.verifier.attempts >= max_tries:
        case.state = CaseState.NEEDS_HUMAN_REVIEW
    else:
        case.state = CaseState.VERIFY_FAILED if notes else CaseState.VERIFIED
    case.log("verifier", f"attempt {case.verifier.attempts}: {case.verifier.status} {notes}", rt.clock.iso())
    return case
