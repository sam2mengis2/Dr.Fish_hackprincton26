"""Agent 5 - Urgency Triage (rules + model scores).

Levels: critical (bypasses reasoning), urgent, routine, incidental.
Case urgency = highest level among findings from the scans just ingested.
"""
from __future__ import annotations

from ..knowledge import LEVEL_ORDER, urgency_for
from ..runtime import Runtime
from ..schema import Case, CaseState


def run(case: Case, rt: Runtime) -> Case:
    new_ids = set(case.pending_scan_ids)
    best, reasons = "incidental", []
    for f in case.findings:
        if f.scan_id not in new_ids:
            continue
        lvl = urgency_for(f.finding, f.size_mm, f.confidence, "report_extraction" in f.sources)
        if f.growth_mm and f.growth_mm >= 2 and LEVEL_ORDER.index(lvl) < LEVEL_ORDER.index("urgent"):
            lvl = "urgent"  # growth over time raises the level
        if LEVEL_ORDER.index(lvl) > LEVEL_ORDER.index(best):
            best, reasons = lvl, [f.finding_id]
        elif lvl == best:
            reasons.append(f.finding_id)
    if not any(f.scan_id in new_ids for f in case.findings):
        best, reasons = "routine", []
    case.urgency = best
    case.urgency_reasons = reasons
    case.pending_scan_ids = []
    case.state = CaseState.TRIAGED
    case.log("triage", f"urgency={best} because {reasons}", rt.clock.iso())
    return case


def worklist_key(case: Case):
    """Sort key for the radiologist worklist: level, then confidence."""
    lvl = LEVEL_ORDER.index(case.urgency) if case.urgency else 0
    conf = max((f.confidence for f in case.findings if f.finding_id in case.urgency_reasons), default=0)
    return (-lvl, -conf)
