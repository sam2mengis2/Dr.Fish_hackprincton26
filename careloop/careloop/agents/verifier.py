"""Agent 7 - Verifier (critic): code checks first, then an LLM critic.

Pass -> Care Planner. Fail -> back to Reasoner with notes (max retries from config),
then NEEDS_HUMAN_REVIEW.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import maybe
from ..runtime import Runtime
from ..schema import Case, CaseState, DxCandidate

import re

UNRELATED = re.compile(r"unrelated|separate|independent|synchronous|second primary|coincident|two (distinct|different)",
                       re.I)


def _similar(a: str, b: str) -> bool:
    words = lambda t: {w for w in re.findall(r"[a-z0-9-]{4,}", t.lower())}
    wa, wb = words(a), words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.3


def ensure_unrelated_option(case: Case, rt: Runtime) -> None:
    """Safety net in code: if any candidate links findings from different body regions, the doctor must also
    see 'these may be separate problems'. LLMs sometimes drop that alternative; this rule never does."""
    region = {s.scan_id: s.body_part.upper() for s in case.scans}
    for d in case.differential:
        regions = {region.get(case.finding(i).scan_id) for i in d.evidence_for if case.finding(i)}
        if len(regions - {None}) >= 2:
            break
    else:
        return
    if any(UNRELATED.search(x.dx) for x in case.differential):
        return
    names = " and ".join(sorted(r.lower() for r in regions if r))
    case.differential.append(DxCandidate(
        rank=len(case.differential) + 1, dx=f"Separate, unrelated processes ({names} findings not connected)",
        likelihood="low", evidence_for=list(d.evidence_for),
        rationale="Added by CareLoop rule: a link across body regions is always weighed against separate causes.",
        caveats=["Rule-added alternative, not AI-ranked."]))
    case.log("verifier", "added 'separate, unrelated processes' alternative (cross-region link rule)", rt.clock.iso())

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
- how growth or stability of a linked lesion ('same lesion as') is recorded: it is ONE observation, cited once
  (latest ID) and described in caveats/likelihood. Never ask for the same finding ID to be listed both for and
  against a candidate, and never treat one ID in both lists as an error if the caveats explain it
- likelihood labels (high/moderate/low) between plausible candidates when the caveats discuss the evidence

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
                             + (f" caveats={d.caveats}" if d.caveats else "")
                             for d in case.differential)
            c = rt.llm.structured("verifier", CRITIC, f"{timeline(case)}\n\nDifferential:\n{diff}", Critique)
            critic, case.verifier.suggestions = c.blocking[:3], c.suggestions[:4]
        else:
            critic = _mock_critic(case)
        # Loop breaker: a critic complaint that repeats an earlier attempt's (the reasoner already tried to fix it
        # and the critic came back to it) becomes a reviewer note for the doctor instead of another retry.
        # Code checks (invented IDs, empty evidence, confidence floor) are never softened.
        repeats = [n for n in critic if any(_similar(n, h) for h in case.verifier.history)]
        if repeats:
            case.verifier.suggestions = (repeats + case.verifier.suggestions)[:5]
            case.log("verifier", f"repeat complaint(s) moved to reviewer notes: {repeats}", rt.clock.iso())
        case.verifier.history += critic
        notes = [n for n in critic if n not in repeats]
    case.verifier.notes = notes
    case.verifier.status = "fail" if notes else "pass"
    if not notes:
        ensure_unrelated_option(case, rt)
    retries = rt.config["pipeline"]["max_reasoner_retries"]
    if case.agent_trace:  # a Deep Agent investigation takes minutes: allow at most one redo
        retries = min(retries, rt.config["pipeline"].get("max_deep_agent_retries", 1))
    max_tries = retries + 1
    if notes and case.verifier.attempts >= max_tries:
        case.state = CaseState.NEEDS_HUMAN_REVIEW
    else:
        case.state = CaseState.VERIFY_FAILED if notes else CaseState.VERIFIED
    case.log("verifier", f"attempt {case.verifier.attempts}: {case.verifier.status} {notes}", rt.clock.iso())
    return case
