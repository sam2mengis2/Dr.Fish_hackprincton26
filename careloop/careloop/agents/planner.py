"""Agent 8 - Care Planner (guideline rules in code; LLM only for wording, optional).

Turns findings + differential into follow-up items with owner, due date and the
rule that produced the date. Intervals come ONLY from knowledge.py, never an LLM.
"""
from __future__ import annotations

from ..knowledge import SERIOUS_WORDS, days_for_urgency, due_from, is_high_risk_lung, lung_nodule_rule
from ..runtime import Runtime
from ..schema import Case, CaseState, FollowUp


def _add(case: Case, action: str, due: str, rule: str, fids: list[str]) -> None:
    if any(fu.action.lower() == action.lower() and fu.status != "completed" for fu in case.followups):
        return
    fu_id = f"FU-{len(case.followups) + 1}"
    owner = case.patient.ordering_doctor
    if "oncology" in action.lower():
        owner = "Oncology team"
    case.followups.append(FollowUp(id=fu_id, action=action, owner=owner, due=due, rule=rule, finding_ids=fids))


def run(case: Case, rt: Runtime) -> Case:
    today = rt.clock.now().date().isoformat()
    latest = max((s.date for s in case.scans), default=today)
    new_scan_findings = [f for f in case.findings if f.date == latest]
    before = len(case.followups)

    if case.urgency == "critical":
        _add(case, "Immediate clinician review", today, "Critical-results rule: notify ordering clinician now",
             case.urgency_reasons)
    # Guideline-driven items for incidental nodules on the newest scans
    big_dx = any(d.likelihood in ("high", "moderate") and "cancer" in d.dx.lower() for d in case.differential)
    for f in new_scan_findings:
        if f.finding == "lung nodule" and f.size_mm and not big_dx:
            stable = bool(f.linked_to) and abs(f.growth_mm or 0) <= 1
            r = lung_nodule_rule(f.size_mm, is_high_risk_lung(case.patient.history, case.patient.age), f.date,
                                 stable_on_followup=stable)
            if r:
                _add(case, r[0], r[1], r[2], [f.finding_id])
    # Next tests of the top candidate
    if case.differential and case.urgency != "critical":
        top = case.differential[0]
        days = days_for_urgency(case.urgency or "routine")
        for t in top.next_tests:
            if "per guideline" in t.lower():
                continue
            _add(case, t, due_from(today, days), f"Top candidate next test; {case.urgency} -> within {days} days",
                 top.evidence_for)

    top_dx = case.differential[0].dx if case.differential else ""
    case.message_tier = "serious" if (case.urgency == "critical" or SERIOUS_WORDS.search(top_dx)) else "routine"
    new_items = case.followups[before:]
    case.plan_summary = "; ".join(f"{fu.action} by {fu.due}" for fu in new_items) or "No new follow-up needed"
    case.state = CaseState.PLANNED
    case.log("planner", f"{len(new_items)} follow-up item(s); tier={case.message_tier}", rt.clock.iso())
    return case
