"""The shared case record: one per patient, the single source of truth.

Every agent reads the Case, does its one job, and writes back ONLY its own fields.
Agents cite findings by finding_id; they never copy finding details.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CaseState(str, Enum):
    INGESTED = "ingested"
    PERCEIVED = "perceived"
    RECONCILED = "reconciled"
    TRIAGED = "triaged"
    REASONED = "reasoned"
    VERIFIED = "verified"
    VERIFY_FAILED = "verify_failed"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    DOCTOR_CALLING = "doctor_calling"
    ACTIVE = "active"
    AWAITING_REPORT = "awaiting_report"   # image acquired, radiologist report not signed yet
    CLOSED = "closed"


Urgency = Literal["critical", "urgent", "routine", "incidental"]
FollowUpStatus = Literal["open", "patient_notified", "booked", "missed", "escalated", "completed", "cancelled"]


class Scan(BaseModel):
    scan_id: str
    modality: str              # DICOM Modality: CR, DX, CT, MR
    body_part: str             # DICOM BodyPartExamined: CHEST, BRAIN, ABDOMEN
    date: str                  # ISO date
    report_text: str = ""
    image_path: Optional[str] = None
    site: str = ""             # hospital/imaging site that performed the scan
    indication: str = ""       # reason for the order, e.g. "ER: flank pain, rule out kidney stone"
    report_status: Literal["pending", "final"] = "final"
    acquired_at: Optional[str] = None   # when the images were taken (sim clock)
    reported_at: Optional[str] = None   # when the radiologist signed the report
    read_priority: Optional[Literal["stat", "urgent", "routine"]] = None
    read_priority_reasons: list[str] = Field(default_factory=list)
    ai_preread: list[dict] = Field(default_factory=list)   # image-model output before the report (queue sorting only)
    outside: bool = False      # imported from another hospital's record (history only, no new workup)
    # Mock-mode stand-in for model output when there is no real image/model.
    mock_model_output: list[dict] = Field(default_factory=list)
    route: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    finding_id: str = ""
    finding: str               # normalized name, e.g. "lung nodule"
    location: str = ""
    size_mm: Optional[float] = None
    count: int = 1
    confidence: float = 0.5
    sources: list[str] = Field(default_factory=list)   # "cxr_model", "report_extraction", ...
    scan_id: str = ""
    date: str = ""
    report_quote: Optional[str] = None
    heatmap_path: Optional[str] = None
    recommendation: Optional[str] = None
    linked_to: Optional[str] = None     # earlier finding_id of the same lesion
    growth_mm: Optional[float] = None


class Discrepancy(BaseModel):
    finding_id: str
    type: Literal["model_only", "report_only"]
    note: str


class DxCandidate(BaseModel):
    rank: int
    dx: str
    likelihood: Literal["high", "moderate", "low"] = "moderate"
    evidence_for: list[str] = Field(default_factory=list,
                                    description="Finding IDs ONLY (e.g. 'F-002') that support this candidate. No prose.")
    evidence_against: list[str] = Field(default_factory=list,
                                        description="Finding IDs ONLY (e.g. 'F-001') that argue against it. No prose.")
    caveats: list[str] = Field(default_factory=list,
                               description="Free-text reasons, missing data and uncertainties (prose goes HERE).")
    rationale: str = ""
    next_tests: list[str] = Field(default_factory=list)


class VerifierResult(BaseModel):
    status: Literal["pass", "fail", "pending"] = "pending"
    notes: list[str] = Field(default_factory=list)          # blocking issues (cause a retry)
    suggestions: list[str] = Field(default_factory=list)    # non-blocking reviewer notes for the doctor
    attempts: int = 0
    history: list[str] = Field(default_factory=list)        # earlier critic complaints (to stop repeat loops)


class FollowUp(BaseModel):
    id: str
    action: str
    owner: str
    due: str                   # ISO date
    rule: str = ""             # which guideline/rule produced the due date
    finding_ids: list[str] = Field(default_factory=list)
    status: FollowUpStatus = "open"
    appointment: Optional[str] = None   # ISO datetime of booked slot
    completed_by_scan: Optional[str] = None
    site: Optional[str] = None          # where the appointment is booked
    waitlisted: bool = False            # booked slot is later than the clinical due date -> cancellation list
    booking_note: str = ""              # why this slot/site (e.g. "earliest of 2 hospitals")
    offer: Optional[dict] = None        # pending earlier-slot offer {site, slot, expires}
    declined: list[str] = Field(default_factory=list)   # offered slots the patient skipped
    days_saved: int = 0                 # days gained by taking freed slots


class Approval(BaseModel):
    by: str
    decision: Literal["APPROVE", "EDIT", "REJECT", "CALL"]
    note: str = ""
    at: str


class Message(BaseModel):
    to: Literal["doctor", "patient", "nurse", "covering_doctor", "radiologist"]
    direction: Literal["out", "in"] = "out"
    tier: Optional[Literal["serious", "routine"]] = None
    text: str
    at: str


class AuditEntry(BaseModel):
    agent: str
    action: str
    at: str


class Timer(BaseModel):
    kind: Literal["approval_reminder", "approval_escalate", "appt_reminder", "appt_missed_check", "rebook_escalate",
                  "report_overdue", "report_overdue_2", "offer_expire"]
    fire_at: str               # ISO datetime on the simulated clock
    ref: str = ""              # follow-up id, if any
    done: bool = False


class Patient(BaseModel):
    patient_id: str
    name: str
    age: int
    sex: str = ""
    history: list[str] = Field(default_factory=list)
    labs: list[str] = Field(default_factory=list)
    meds: list[str] = Field(default_factory=list)
    ordering_doctor: str = "Dr. Lee"
    preferred_site: str = ""            # where the patient would rather go for follow-ups (e.g. closer to home)
    demo_auto_reply: bool = False       # simulated waitlist patient: accepts earlier-slot offers automatically
    language: str = "en"


class Case(BaseModel):
    patient: Patient
    state: CaseState = CaseState.INGESTED
    scans: list[Scan] = Field(default_factory=list)
    pending_scan_ids: list[str] = Field(default_factory=list)   # scans not yet perceived
    raw_findings: list[Finding] = Field(default_factory=list)   # pre-reconciliation, current batch
    findings: list[Finding] = Field(default_factory=list)
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    urgency: Optional[Urgency] = None
    urgency_reasons: list[str] = Field(default_factory=list)
    differential: list[DxCandidate] = Field(default_factory=list)
    insufficient_info: bool = False
    # Deep Agent investigator output (shown on the dashboard)
    agent_plan: list[str] = Field(default_factory=list)
    workup_note: str = ""
    agent_llm_calls: int = 0            # model calls made inside the Deep Agent (incl. subagents)
    agent_trace: list[str] = Field(default_factory=list)
    verifier: VerifierResult = Field(default_factory=VerifierResult)
    followups: list[FollowUp] = Field(default_factory=list)
    plan_summary: str = ""
    message_tier: Optional[Literal["serious", "routine"]] = None
    approvals: list[Approval] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    timers: list[Timer] = Field(default_factory=list)
    audit_log: list[AuditEntry] = Field(default_factory=list)

    # ---- helpers -------------------------------------------------------
    def log(self, agent: str, action: str, at: str) -> None:
        self.audit_log.append(AuditEntry(agent=agent, action=action, at=at))

    def finding(self, fid: str) -> Optional[Finding]:
        return next((f for f in self.findings if f.finding_id == fid), None)

    def next_finding_id(self) -> str:
        n = len(self.findings) + 1
        existing = {f.finding_id for f in self.findings}
        while f"F-{n:03d}" in existing:
            n += 1
        return f"F-{n:03d}"

    def followup(self, fu_id: str) -> Optional[FollowUp]:
        return next((f for f in self.followups if f.id == fu_id), None)


_FID = __import__("re").compile(r"^F-\d{3,}$")
_FID_ANY = __import__("re").compile(r"\bF-\d{3,}\b")


def tidy_differential(cands: list["DxCandidate"]) -> list["DxCandidate"]:
    """LLMs sometimes write reasons into the ID lists. Keep only ID-shaped entries there and
    move the prose to caveats. Invented IDs (e.g. F-999) still look like IDs, so the Verifier
    still catches them; a candidate left with no evidence_for is flagged by the Verifier too."""
    for d in cands:
        for field in ("evidence_for", "evidence_against"):
            ids, prose = [], []
            for x in getattr(d, field):
                found = _FID_ANY.findall(x)
                ids.extend(i for i in found if i not in ids)
                rest = _FID_ANY.sub("", x).strip(" ,;:-()[]")
                if rest:
                    prose.append(x.strip())
            setattr(d, field, ids)
            d.caveats.extend(prose)
    return cands


def now_iso(dt: datetime) -> str:
    return dt.isoformat(timespec="minutes")
