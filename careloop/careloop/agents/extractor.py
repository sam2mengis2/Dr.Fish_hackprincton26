"""Agent 3 - Report Extractor (LLM, structured output).

Turns free-text radiology reports into findings + recommendations. Works for ANY
scan type, which is what makes CareLoop general. Guard: every report_quote must
appear verbatim in the report (checked in code) or the finding is dropped.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from ..knowledge import SYNONYMS, normalize
from ..llm import maybe
from ..runtime import Runtime
from ..schema import Case, Finding, Scan


class ExtractedFinding(BaseModel):
    finding: str = Field(description="short name, e.g. 'lung nodule', 'pneumothorax'")
    location: str = ""
    size_mm: Optional[float] = None
    count: int = 1
    hedged: bool = Field(False, description="true if 'possible', 'cannot exclude', 'may represent'")
    report_quote: str = Field(description="the exact sentence from the report, copied verbatim")
    recommendation: Optional[str] = None


class ExtractionOut(BaseModel):
    findings: list[ExtractedFinding]


SYSTEM = """You extract findings from radiology reports.
Rules:
- Extract EVERY abnormal finding, including incidental ones mentioned in passing.
- Negated findings ("no pneumothorax", "without effusion") are NOT findings.
- report_quote must be copied verbatim from the report, one sentence.
- Convert cm to mm. Mark hedged=true for 'possible', 'cannot exclude', 'may represent', 'suspicious for'.
- Put any follow-up recommendation the radiologist gives in `recommendation`."""

NEGATION = re.compile(r"\b(no|without|negative for|free of|resolved)\b", re.I)
HEDGE = re.compile(r"\b(possible|possibly|cannot (be )?exclude|may represent|suspicious for|questionable)\b", re.I)
SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", re.I)
COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "multiple": 3, "several": 3}
LOCATION = re.compile(
    r"\b((?:right|left|bilateral)\s+(?:upper|middle|lower)?\s*(?:lobe|hilum|hilar region|apex|base|"
    r"frontal lobe|parietal lobe|temporal lobe|occipital lobe|frontal convexity|kidney|ureter|hemithorax)"
    r"|(?:right|left)\s+\w+\s+lobe|(?:right|left)\s+hilar)",
    re.I,
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _mock_extract(report: str) -> list[ExtractedFinding]:
    out: list[ExtractedFinding] = []
    sents = _sentences(report)
    recs = [s for s in sents if re.search(r"\brecommend", s, re.I)]
    for s in sents:
        if re.search(r"\brecommend", s, re.I):
            continue
        low = s.lower()
        # skip clauses that are negated before the term
        for key in sorted(SYNONYMS, key=len, reverse=True):
            idx = low.find(key)
            if idx < 0:
                continue
            prefix = low[max(0, idx - 30):idx]
            if NEGATION.search(prefix):
                break
            size = None
            m = SIZE.search(s)
            if m:
                size = float(m.group(1)) * (10 if m.group(2).lower() == "cm" else 1)
            count = 1
            cm = re.search(r"\b(" + "|".join(COUNT_WORDS) + r"|\d+)\s+(?:\w+[- ])*?(?:lesions|nodules|masses)", low)
            if cm:
                w = cm.group(1)
                count = COUNT_WORDS.get(w, int(w) if w.isdigit() else 1)
            loc = LOCATION.search(s)
            if loc:  # one spelling per place, so locations match across scans ("right hilar" -> "right hilum")
                loc_text = loc.group(1).lower().replace("hilar region", "hilum").replace("hilar", "hilum")
            canon = normalize(key)
            rec = next((r for r in recs if canon.split()[-1] in r.lower() or "ct" in r.lower()), None)
            out.append(ExtractedFinding(
                finding=canon, location=loc_text if loc else "", size_mm=size, count=count,
                hedged=bool(HEDGE.search(s)), report_quote=s, recommendation=rec,
            ))
            break  # one finding per sentence in mock mode
    return out


def _llm_extract(report: str, scan: Scan, rt: Runtime) -> list[ExtractedFinding]:
    user = f"Scan type: {scan.modality} {scan.body_part}\n\nReport:\n{report}"
    return rt.llm.structured("extractor", SYSTEM, user, ExtractionOut).findings


def run(case: Case, rt: Runtime) -> Case:
    for sid in case.pending_scan_ids:
        scan = next(s for s in case.scans if s.scan_id == sid)
        if "report_extraction" not in scan.route:
            continue
        raw = _llm_extract(scan.report_text, scan, rt) if maybe(rt.llm) else _mock_extract(scan.report_text)
        kept = 0
        for ef in raw:
            if ef.report_quote not in scan.report_text:  # verbatim guard against hallucinated findings
                case.log("extractor", f"DROPPED non-verbatim finding '{ef.finding}'", rt.clock.iso())
                continue
            case.raw_findings.append(Finding(
                finding=normalize(ef.finding), location=ef.location, size_mm=ef.size_mm, count=ef.count,
                confidence=0.6 if ef.hedged else 0.9, sources=["report_extraction"], scan_id=sid,
                date=scan.date, report_quote=ef.report_quote, recommendation=ef.recommendation,
            ))
            kept += 1
        case.log("extractor", f"{sid}: {kept} finding(s) from report", rt.clock.iso())
    return case
