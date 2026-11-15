"""Agent 8 - Care Planner (guideline rules in code; LLM only for wording, optional).

Turns findings + differential into follow-up items with owner, due date and the
rule that produced the date. Intervals come ONLY from knowledge.py, never an LLM.
"""
from __future__ import annotations

from ..knowledge import SERIOUS_WORDS, clean_action, is_clinician_task, test_type, days_for_urgency, due_from, is_high_risk_lung, lung_nodule_rule
import re

from ..runtime import Runtime

CONDITIONAL = re.compile(r"\b(if|when|should|consider|unless|as needed|prn)\b|\breview (ct|imaging|images)\b", re.I)
from ..schema import Case, CaseState, FollowUp


def _add(case: Case, action: str, due: str, rule: str, fids: list[str], urgent: bool = False) -> None:
    action = clean_action(action)
    if is_clinician_task(action):
        rule = rule + " | clinician task (not booked, not sent to patient)"
    kind = test_type(action)
    pending = [fu for fu in case.followups if test_type(fu.action) == kind and fu.status != "completed"]
    if pending:
        fu = pending[0]
        guideline_set = "Fleischner" in fu.rule or "guideline" in fu.rule.lower()
        if due < fu.due and guideline_set and not urgent:
            case.log("planner", f"AI suggested '{action}' by {due}; guideline interval for {fu.id} kept ({fu.due})", "")
        elif due < fu.due:  # same test now needed sooner: move the existing item earlier and rebook it
            fu.due, fu.rule, fu.appointment, fu.status = due, rule + f" (moved earlier from {fu.due})", None, "open"
            for t in case.timers:
                if t.ref == fu.id:
                    t.done = True
            case.log("planner", f"{fu.id} '{fu.action}' moved earlier to {due} ({kind} needed sooner)", "")
        else:
            case.log("planner", f"skipped '{action}': a {kind} test is already pending ({fu.id})", "")
        return
    fu_id = f"FU-{len(case.followups) + 1}"
    owner = case.patient.ordering_doctor
    if "oncology" in action.lower():
        owner = "Oncology team"
    if is_clinician_task(action):
        owner = case.patient.ordering_doctor
    case.followups.append(FollowUp(id=fu_id, action=action, owner=owner, due=due, rule=rule, finding_ids=fids))


def run(case: Case, rt: Runtime) -> Case:
    today = rt.clock.now().date().isoformat()
    latest = max((s.date for s in case.scans), default=today)
    new_scan_findings = [f for f in case.findings if f.date == latest]
    before = len(case.followups)
    review = case.state == CaseState.NEEDS_HUMAN_REVIEW  # AI reasoning not trusted: rules-only plan

    if case.urgency == "critical":
        _add(case, "Immediate clinician review", today, "Critical-results rule: notify ordering clinician now",
             case.urgency_reasons)
    # Guideline-driven items for incidental nodules on the newest scans
    for f in new_scan_findings:
        if f.finding == "lung nodule" and f.size_mm:  # guideline always applies; AI likelihood can't switch it off
            stable = bool(f.linked_to) and abs(f.growth_mm or 0) <= 1
            r = lung_nodule_rule(f.size_mm, is_high_risk_lung(case.patient.history, case.patient.age), f.date,
                                 stable_on_followup=stable, growth_mm=f.growth_mm if f.linked_to else None)
            if r:
                _add(case, r[0], r[1], r[2], [f.finding_id])
    # Next tests of the top candidate
    if case.differential and case.urgency != "critical" and not review:
        new_ids = {f.finding_id for f in new_scan_findings}
        # Prefer the highest-ranked candidate that explains a finding from the newest scan, so a
        # follow-up scan for the nodule doesn't restart workup for an older, unrelated problem.
        top = next((d for d in case.differential if set(d.evidence_for) & new_ids), case.differential[0])
        days = days_for_urgency(case.urgency or "routine")
        concrete = [t for t in top.next_tests
                    if "per guideline" not in t.lower()
                    and not CONDITIONAL.search(t)]  # "X if Y" is a contingency, not an appointment
        for t in concrete[:3]:
            _add(case, t, due_from(today, days), f"Top candidate next test; {case.urgency} -> within {days} days",
                 top.evidence_for, urgent=case.urgency in ("urgent", "critical"))

    top_dx = case.differential[0].dx if case.differential else ""
    case.message_tier = "serious" if (case.urgency == "critical" or SERIOUS_WORDS.search(top_dx)) else "routine"
    new_items = case.followups[before:]
    case.plan_summary = "; ".join(f"{fu.action} by {fu.due}" for fu in new_items) or "No new follow-up needed"
    case.state = CaseState.NEEDS_HUMAN_REVIEW if review else CaseState.PLANNED
    case.log("planner", f"{len(new_items)} follow-up item(s); tier={case.message_tier}", rt.clock.iso())
    return case
