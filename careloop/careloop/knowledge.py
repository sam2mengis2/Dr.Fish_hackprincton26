"""Vocabulary normalization, urgency rules and follow-up guideline rules.

IMPORTANT: these are SIMPLIFIED DEMO RULES for a hackathon prototype. Before showing
them as clinical logic, check them against the published sources (Fleischner Society
2017 guidelines for incidental pulmonary nodules; ACR incidental findings white papers;
your target hospital's critical-results policy). Intervals come only from code here,
never from an LLM.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

# ---------------------------------------------------------------- vocabulary
# Map synonyms -> canonical finding names (stand-in for RadLex/SNOMED coding).
SYNONYMS: dict[str, str] = {
    "nodule": "lung nodule",
    "pulmonary nodule": "lung nodule",
    "lung nodule": "lung nodule",
    "small opacity": "lung nodule",
    "density": "lung nodule",
    "mass": "lung mass",
    "lung mass": "lung mass",
    "hilar mass": "lung mass",
    "pulmonary mass": "lung mass",
    "pneumothorax": "pneumothorax",
    "effusion": "pleural effusion",
    "pleural effusion": "pleural effusion",
    "consolidation": "consolidation",
    "pneumonia": "consolidation",
    "airspace opacity": "consolidation",
    "infiltration": "consolidation",
    "cardiomegaly": "cardiomegaly",
    "enlarged cardiac silhouette": "cardiomegaly",
    "atelectasis": "atelectasis",
    "ring-enhancing lesion": "ring-enhancing brain lesion",
    "ring enhancing lesion": "ring-enhancing brain lesion",
    "ring-enhancing lesions": "ring-enhancing brain lesion",
    "brain lesion": "ring-enhancing brain lesion",
    "meningioma": "extra-axial brain mass",
    "extra-axial mass": "extra-axial brain mass",
    "midline shift": "midline shift",
    "kidney stone": "kidney stone",
    "renal calculus": "kidney stone",
    "nephrolithiasis": "kidney stone",
    "ureteral stone": "kidney stone",
    "ureteral calculus": "kidney stone",
    "ureterolithiasis": "kidney stone",
    "urolithiasis": "kidney stone",
    "free air": "pneumoperitoneum",
    "pneumoperitoneum": "pneumoperitoneum",
}

# TorchXRayVision pathology label -> canonical name
TXRV_TO_CANON = {
    "Pneumothorax": "pneumothorax",
    "Mass": "lung mass",
    "Nodule": "lung nodule",
    "Effusion": "pleural effusion",
    "Consolidation": "consolidation",
    "Pneumonia": "consolidation",
    "Cardiomegaly": "cardiomegaly",
    "Atelectasis": "atelectasis",
    "Lung Lesion": "lung nodule",
}


CANONICAL = set(SYNONYMS.values())


def normalize(name: str) -> str:
    n = name.strip().lower()
    if n in CANONICAL:
        return n
    if n in SYNONYMS:
        return SYNONYMS[n]
    for k in sorted(SYNONYMS, key=len, reverse=True):
        if k in n:
            return SYNONYMS[k]
    return n


# ---------------------------------------------------------------- urgency
CRITICAL = {"pneumothorax", "midline shift", "pneumoperitoneum"}
URGENT = {"lung mass", "ring-enhancing brain lesion"}
ROUTINE = {"consolidation", "pleural effusion", "cardiomegaly", "atelectasis", "kidney stone"}
INCIDENTAL = {"lung nodule", "extra-axial brain mass"}

LEVEL_ORDER = ["incidental", "routine", "urgent", "critical"]


def urgency_for(finding: str, size_mm: Optional[float], confidence: float, report_confirmed: bool) -> str:
    if finding in CRITICAL:
        # Guard against alarm fatigue: critical needs high confidence OR report confirmation.
        return "critical" if (confidence >= 0.8 or report_confirmed) else "urgent"
    if finding in URGENT:
        return "urgent"
    if finding == "lung nodule" and size_mm and size_mm > 8:
        return "urgent"
    if finding in INCIDENTAL:
        return "incidental"
    return "routine"


# ---------------------------------------------------------------- follow-up rules
def is_high_risk_lung(history: list[str], age: int) -> bool:
    text = " ".join(history).lower()
    return bool(re.search(r"smok|tobacco|pack-year|asbestos|lung cancer", text)) or age >= 65


def lung_nodule_rule(size_mm: float, high_risk: bool, scan_date: str,
                     stable_on_followup: bool = False, growth_mm: Optional[float] = None) -> Optional[tuple[str, str, str]]:
    """Simplified Fleischner 2017, single solid incidental nodule.
    Returns (action, due_iso, rule_text) or None if no routine follow-up."""
    d = date.fromisoformat(scan_date)
    if growth_mm is not None and growth_mm >= 2:
        return ("Short-interval chest CT (growing nodule)", (d + timedelta(days=90)).isoformat(),
                f"Fleischner 2017 (simplified): solid nodule grew {growth_mm:g} mm since prior CT -> suspicious; "
                "CT within 3 months, consider PET-CT or sampling")
    if stable_on_followup and 6 <= size_mm <= 8:
        return ("Follow-up chest CT (second interval)", (d + timedelta(days=365)).isoformat(),
                "Fleischner 2017 (simplified): 6-8 mm stable at first follow-up -> consider CT at 18-24 months")
    if size_mm < 6:
        if high_risk:
            return ("Optional follow-up chest CT", (d + timedelta(days=365)).isoformat(),
                    "Fleischner 2017 (simplified): solid <6 mm, high risk -> optional CT at 12 months")
        return None
    if size_mm <= 8:
        return ("Follow-up chest CT", (d + timedelta(days=182)).isoformat(),
                "Fleischner 2017 (simplified): solid 6-8 mm -> CT at 6-12 months")
    return ("Chest CT, or PET-CT / tissue sampling", (d + timedelta(days=90)).isoformat(),
            "Fleischner 2017 (simplified): solid >8 mm -> consider CT at 3 months, PET-CT or sampling")


_TEST_TYPES = [
    ("biopsy", re.compile(r"biops|ebus|tbna|bronchoscop|tissue sampling|fine.needle|\bfna\b", re.I)),
    ("pet-ct", re.compile(r"\bpet\b", re.I)),
    ("ct-chest", re.compile(r"(\bct\b|computed tomography).*(chest|thorax)|(chest|thorax).*\bct\b", re.I)),
    ("mri-brain", re.compile(r"\bmri\b.*brain|brain.*\bmri\b", re.I)),
    ("radiation-oncology", re.compile(r"radiation oncolog|radiotherap|radiosurg|\bsrs\b", re.I)),
    ("oncology", re.compile(r"oncolog", re.I)),
    ("neurosurgery", re.compile(r"neurosurg", re.I)),
    ("echo", re.compile(r"echocardiogra", re.I)),
    ("urinalysis", re.compile(r"urinalysis|\bua\b", re.I)),
    ("renal-function", re.compile(r"creatinine|metabolic panel|\bbmp\b|renal function", re.I)),
]


def test_type(action: str) -> str:
    """Coarse test category so 'CT chest/abdomen with contrast' and 'Contrast-enhanced CT chest' count as
    the same pending test. Falls back to the lowercased text for anything uncategorized."""
    for key, rx in _TEST_TYPES:
        if rx.search(action):
            return key
    return action.strip().lower()


_COMMENTARY = re.compile(
    r"\s*\((?=[^)]*(coded|judg|decision|clinician|optional|if indicated|per mdt|per team))[^)]*\)"
    r"|\s*[-\u2013\u2014]\s*(clinician|physician|team|mdt)[^,;]*(decision|discretion|judg\w*)\s*$", re.I)
CLINICIAN_TASK = re.compile(
    r"\b(confirm|verify|check whether|review|discuss|input|consult with|tumou?r board|multidisciplinary|\bmdt\b|"
    r"correlat\w*|compare with prior)\b", re.I)


def clean_action(action: str) -> str:
    """Strip the model's commentary from a test name: '(timeframe is clinical judgment, not a coded rule)',
    '- clinician decision'. What's left is what goes on the plan and the calendar."""
    return re.sub(r"\s{2,}", " ", _COMMENTARY.sub("", action)).strip(" -,;")


def is_clinician_task(action: str) -> bool:
    """Work for the care team (confirm/review/discuss/MDT), not an appointment the patient attends."""
    return bool(CLINICIAN_TASK.search(action))


def days_for_urgency(level: str) -> int:
    return {"critical": 1, "urgent": 7, "routine": 30, "incidental": 180}.get(level, 30)


def due_from(scan_date: str, days: int) -> str:
    return (date.fromisoformat(scan_date) + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------- patient-safe wording
SERIOUS_WORDS = re.compile(
    r"\b(cancer|malignan\w*|metasta\w*|tumou?r|carcinoma|lymphoma|terminal|abscess|oncolog\w*)\b", re.I
)


def patient_label(action: str) -> str:
    """Patient-facing name for a follow-up. Use this in EVERY patient text (reminders, missed, rebook)."""
    a = action.lower()
    if "oncolog" in a or "referral" in a or "consult" in a or "neurosurg" in a:
        return "specialist appointment"
    if re.search(r"biops|sampling|ebus|tbna|bronchoscop", a):
        return "procedure"
    plain = re.sub(r"\s*\([^)]*\)", "", action).strip()   # "MRI brain (metastasis protocol)" -> "MRI brain"
    if not SERIOUS_WORDS.search(plain):
        return plain
    friendly = {"mri-brain": "brain MRI", "ct-chest": "chest CT", "pet-ct": "PET-CT scan"}
    return friendly.get(test_type(action), "appointment")
