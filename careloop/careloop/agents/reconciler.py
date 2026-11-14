"""Agent 4 - Reconciler (code first).

Merges model findings and report findings into one list with stable IDs,
flags disagreements, and links the same lesion across scans over time.
"""
from __future__ import annotations

from ..runtime import Runtime
from ..schema import Case, CaseState, Discrepancy, Finding

# Findings that count as "the same lesion" across scans
LINKABLE = {"lung nodule": {"lung nodule", "lung mass"}, "lung mass": {"lung nodule", "lung mass"}}


def _side(loc: str) -> str:
    loc = loc.lower()
    has_r, has_l = "right" in loc, "left" in loc
    if (has_r and has_l) or "bilateral" in loc:
        return ""  # multiple sides: compatible with either ("left frontal and right parietal")
    return "right" if has_r else "left" if has_l else ""


def _same(a: Finding, b: Finding) -> bool:
    if a.scan_id != b.scan_id or a.finding != b.finding:
        return False
    sa, sb = _side(a.location), _side(b.location)
    return not sa or not sb or sa == sb


def run(case: Case, rt: Runtime) -> Case:
    now = rt.clock.iso()
    model = [f for f in case.raw_findings if "report_extraction" not in f.sources]
    report = [f for f in case.raw_findings if "report_extraction" in f.sources]
    merged: list[Finding] = []
    used = set()
    for r in report:
        match = next((m for i, m in enumerate(model) if i not in used and _same(m, r)), None)
        if match is not None:
            used.add(model.index(match))
            r.sources = sorted(set(r.sources + match.sources))
            r.confidence = round(min(0.99, max(r.confidence, match.confidence) + 0.05), 2)
            r.heatmap_path = match.heatmap_path
            r.location = r.location or match.location
            r.size_mm = r.size_mm or match.size_mm
        merged.append(r)  # report-only findings are kept: the radiologist is the reference
    model_only = [m for i, m in enumerate(model) if i not in used]

    for f in merged + model_only:
        f.finding_id = case.next_finding_id()
        # cross-scan linking: same lesion seen on an earlier scan?
        prior = [p for p in case.findings
                 if p.finding in LINKABLE.get(f.finding, {f.finding}) and p.scan_id != f.scan_id
                 and (not _side(p.location) or not _side(f.location) or _side(p.location) == _side(f.location))]
        if prior:
            p = prior[-1]
            f.linked_to = p.finding_id
            if p.size_mm and f.size_mm:
                f.growth_mm = round(f.size_mm - p.size_mm, 1)
        case.findings.append(f)

    for m in model_only:
        case.discrepancies.append(Discrepancy(
            finding_id=m.finding_id, type="model_only",
            note=f"model flags {m.finding} ({m.confidence}); report does not mention it - second look",
        ))
    case.raw_findings = []
    case.state = CaseState.RECONCILED
    case.log("reconciler", f"{len(merged)} confirmed/report findings, {len(model_only)} model-only discrepancies",
             now)
    return case
