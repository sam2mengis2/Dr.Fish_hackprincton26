"""Agent 2b - Reading-queue prioritizer (rules first; LLM may only raise priority).

Runs the moment the images exist, before any radiologist report. Ranks the scan in the
radiologist's reading queue as stat / urgent / routine using:
  1. the reason for the order (indication text), via keyword rules
  2. open follow-ups this scan answers (e.g. a growing nodule on a 3-month deadline)
  3. the image model's pre-read, where a real model ran (chest X-ray today)
  4. optionally an LLM read of the indication, which can raise but never lower the rule level
The pre-read only sorts the queue. It is never sent to the family doctor as a result.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from ..llm import maybe
from ..runtime import Runtime
from ..schema import Case, Scan

LEVELS = ["routine", "urgent", "stat"]

STAT = re.compile(r"\b(stroke|code|trauma|pneumothorax|pulmonary embol|\bPE\b|hemorrhage|haemorrhage|bleed|"
                  r"dissection|cord compression|unresponsive|seizure|collapse|stat)\b", re.I)
URGENT = re.compile(r"\b(mass|cancer|malignan|metasta|staging|tumou?r|growing|grew|short-interval|suspicious|"
                    r"weight loss|hemoptysis|new deficit|abscess|obstruct|appendicitis)\b", re.I)
CRITICAL_LABELS = {"pneumothorax", "mass", "hemorrhage", "pneumoperitoneum"}

SYSTEM = """You sort a radiologist's reading queue. Given the reason a scan was ordered, answer how fast it
must be read: stat (minutes-hours, life-threatening possibility), urgent (within a day), or routine.
Give a short reason in plain words. Do not diagnose."""


class _Prio(BaseModel):
    priority: str
    reason: str


def _max(a: str, b: str) -> str:
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


def prioritize(case: Case, scan: Scan, rt: Runtime) -> Scan:
    level, reasons = "routine", []
    ind = scan.indication or ""
    m = STAT.search(ind)
    if m:
        level, reasons = "stat", [f"order reason mentions '{m.group(0)}'"]
    else:
        m = URGENT.search(ind)
        if m:
            level, reasons = "urgent", [f"order reason mentions '{m.group(0)}'"]

    # A scan that answers an open follow-up inherits that follow-up's urgency.
    for fu in case.followups:
        if fu.status != "completed" and ("grow" in fu.rule.lower() or "urgent" in fu.rule.lower()
                                         or "critical" in fu.rule.lower()):
            level = _max(level, "urgent")
            reasons.append(f"answers open follow-up {fu.id} ({fu.action}, due {fu.due})")
            break

    # Image-model pre-read (only where a real model ran on a real image, or mock output in tests/demo).
    for o in scan.ai_preread:
        label, prob = str(o.get("label", "")).lower(), float(o.get("prob", 0))
        if label in CRITICAL_LABELS and prob >= 0.8:
            level = _max(level, "stat")
            reasons.append(f"image model flags {label} ({prob:.2f}) - for read order only")
        elif prob >= 0.6:
            level = _max(level, "urgent" if label in CRITICAL_LABELS else level)
            if label in CRITICAL_LABELS:
                reasons.append(f"image model flags {label} ({prob:.2f}) - for read order only")

    if maybe(rt.llm) and ind:
        try:
            out = rt.llm.structured("comms", SYSTEM, f"Reason for order: {ind}", _Prio)
            llm_level = out.priority.strip().lower()
            if llm_level in LEVELS and LEVELS.index(llm_level) > LEVELS.index(level):
                level = llm_level  # the LLM can only move a scan UP the queue
                reasons.append(f"AI read of order reason: {out.reason}")
        except Exception as e:  # never block the queue on an LLM error
            case.log("read_priority", f"LLM priority skipped: {e}", rt.clock.iso())

    scan.read_priority = level
    scan.read_priority_reasons = reasons or ["no urgent signals; first-come order"]
    case.log("read_priority", f"{scan.scan_id} queued {level}: {'; '.join(scan.read_priority_reasons)}",
             rt.clock.iso())
    return scan
