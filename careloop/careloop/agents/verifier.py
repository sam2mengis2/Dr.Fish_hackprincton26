"""Agent 7 - Verifier (critic): code checks first, then an LLM critic.

Pass -> Care Planner. Fail -> back to Reasoner with notes (max retries from config),
then NEEDS_HUMAN_REVIEW.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import maybe
from ..runtime import Runtime
from ..schema import Case, CaseState

CRITIC = """You are an attending radiologist reviewing an AI's differential diagnosis BEFORE a physician sees it.
Judge it against the data that EXISTS in the timeline. You are a quality gate, not a perfectionist.

BLOCKING issues (put in `blocking`; any one fails the review):
- a cited finding does not support the candidate it is cited for, or is cited for an unrelated organ/claim
- the same lesion over time (linked findings) is counted as independent evidence to inflate support
- the ranking clearly contradicts the evidence (e.g. a low-evidence candidate ranked above a well-supported one)
- an obvious, common explanation for a finding is missing entirely
- a dangerous possibility that the findings plainly raise is missing entirely

NOT blocking (put in `suggestions` instead):
- data that is simply not in the record (smoking history, morphology, prior imaging, labs)
- wording, extra detail, or nice-to-have workup
- disagreements about ordering between candidates of similar likelihood
- an 'unrelated findings' or alternative candidate citing the findings it is trying to explain (that is correct)

Return at most 3 blocking issues and at most 4 suggestions, each one sentence. If there are no blocking
issues, return blocking=[] even if you have suggestions."""


class Critique(BaseModel):
    blocking: list[str] = Field(default_factory=list, description="Issues that must be fixed before a doctor sees it")
    suggestions: list[str] = Field(default_factory=list, description="Non-blocking notes, e.g. missing data")


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
        if not d.evidence_for:
            notes.append(f"'{d.dx}' cites no finding IDs in evidence_for")
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
            notes = c.blocking[:3]
            case.verifier.suggestions = c.suggestions[:4]
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
