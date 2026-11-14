"""Agent 6 - Diagnostic Reasoner (LLM).

Builds a compact patient timeline in code and asks for 3-5 ranked possible
diagnoses. Hard rules: cite a finding ID for every claim, include at least one
benign explanation, consider that findings may be unrelated, and say
'insufficient information' when warranted. On Verifier failure it reruns with
the Verifier's notes.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import maybe
from ..runtime import Runtime
from ..schema import Case, CaseState, DxCandidate, tidy_differential

SYSTEM = """You are a diagnostic reasoning assistant supporting (never replacing) a physician.
Given a patient's imaging findings timeline and history, produce 3-5 ranked POSSIBLE diagnoses.
Rules:
1. evidence_for / evidence_against contain finding IDs ONLY (e.g. F-004), never sentences; never cite an ID
   not in the timeline. Put reasons, missing data and uncertainties in `caveats`.
2. Include at least one benign or alternative explanation.
3. Explicitly consider that findings may be UNRELATED; if so, include a candidate saying so.
4. Do not lean on findings with confidence below {floor} unless you say they are weak.
5. If the findings do not support any conclusion, set insufficient_info=true.
6. next_tests: at most 3 concrete, orderable tests that would confirm or rule out the candidate (e.g. 'Urinalysis',
   'PET-CT'). No conditional items ('X if Y') and no tasks for the radiologist; put contingencies in caveats.
   Only suggest tests that are NOT already in the current follow-up plan.
7. The same lesion seen on several scans (marked 'same lesion as') is ONE observation: cite the most recent ID
   and describe the change over time in caveats. Never cite a finding for a claim about a different organ.
8. Prefer specific diagnoses. If findings in different organ systems have no plausible shared cause, say so in
   caveats and give per-finding candidates; add an explicit 'unrelated' candidate only when you also propose a link.
   An 'unrelated findings' candidate lists the findings it covers in evidence_for (they are what it explains).
9. Data that is not in the record (smoking, morphology, prior imaging) goes in caveats; set likelihood from the
   evidence that IS present.
This is decision support for a doctor, not a diagnosis."""


class ReasonerOut(BaseModel):
    insufficient_info: bool = False
    differential: list[DxCandidate] = Field(default_factory=list)


def timeline(case: Case) -> str:
    p = case.patient
    lines = [f"Patient {p.patient_id}, {p.age}{p.sex}. History: {'; '.join(p.history) or 'none'}. "
             f"Labs: {'; '.join(p.labs) or 'none'}. Meds: {'; '.join(p.meds) or 'none'}.", "Findings by date:"]
    for f in sorted(case.findings, key=lambda x: (x.date, x.finding_id)):
        extra = []
        if f.size_mm:
            extra.append(f"{f.size_mm:g} mm")
        if f.count > 1:
            extra.append(f"x{f.count}")
        if f.linked_to:
            extra.append(f"same lesion as {f.linked_to}, growth {f.growth_mm} mm")
        lines.append(f"- {f.date} {f.finding_id}: {f.finding} {f.location} ({', '.join(extra)}) "
                     f"conf={f.confidence} sources={'+'.join(f.sources)} scan={f.scan_id}")
    open_items = [fu for fu in case.followups if fu.status != "completed"]
    if open_items:
        lines.append("Current follow-up plan (already ordered; do NOT re-suggest these or equivalents):")
        lines += [f"- {fu.action} ({fu.status}, due {fu.due})" for fu in open_items]
    return "\n".join(lines)


def _ids(case: Case, *names: str) -> list[str]:
    # Model-only findings below 0.8 stay discrepancies; they don't drive the mock reasoning.
    return [f.finding_id for f in case.findings
            if f.finding in names and ("report_extraction" in f.sources or f.confidence >= 0.8)]


def _mock_reason(case: Case, rt: Runtime) -> ReasonerOut:
    mass = _ids(case, "lung mass")
    brain = _ids(case, "ring-enhancing brain lesion")
    nodule = _ids(case, "lung nodule")
    ptx = _ids(case, "pneumothorax")
    cardio = _ids(case, "cardiomegaly")
    meningioma = _ids(case, "extra-axial brain mass")
    d: list[DxCandidate] = []
    if mass and brain:
        d = [
            DxCandidate(rank=1, dx="Primary lung cancer with brain metastases", likelihood="high",
                        evidence_for=mass + brain, next_tests=["PET-CT", "Oncology referral", "Tissue biopsy of lung mass"],
                        rationale="Lung mass followed by multiple ring-enhancing brain lesions in a high-risk patient."),
            DxCandidate(rank=2, dx="Brain abscesses with a lung source of infection", likelihood="low",
                        evidence_for=brain + mass, next_tests=["Blood cultures", "CBC and CRP"],
                        rationale="Ring enhancement can also be infectious; benign alternative."),
            DxCandidate(rank=3, dx="Two unrelated processes (lung lesion and separate brain pathology)",
                        likelihood="low", evidence_for=mass + brain, next_tests=["Neuro-oncology review"],
                        rationale="Findings may be unrelated."),
        ]
    elif mass:
        d = [
            DxCandidate(rank=1, dx="Possible primary lung cancer", likelihood="moderate", evidence_for=mass,
                        next_tests=["Chest CT with contrast", "PET-CT"], rationale="New lung mass."),
            DxCandidate(rank=2, dx="Infection or granuloma", likelihood="low", evidence_for=mass,
                        next_tests=["Chest CT with contrast"], rationale="Benign alternative."),
        ]
    elif ptx:
        d = [DxCandidate(rank=1, dx="Pneumothorax requiring urgent assessment", likelihood="high",
                         evidence_for=ptx, next_tests=["Immediate clinician review", "Repeat chest X-ray"],
                         rationale="Pneumothorax on imaging.")]
    elif cardio and meningioma:
        d = [
            DxCandidate(rank=1, dx="Two unrelated findings: cardiomegaly and an incidental extra-axial mass",
                        likelihood="high", evidence_for=cardio + meningioma,
                        next_tests=["Echocardiogram", "Brain MRI follow-up"],
                        rationale="Different organ systems, no shared mechanism; manage separately."),
            DxCandidate(rank=2, dx="Cardiomegaly from heart failure or valve disease", likelihood="moderate",
                        evidence_for=cardio, next_tests=["Echocardiogram", "BNP"], rationale="Cardiac workup."),
            DxCandidate(rank=3, dx="Likely meningioma, benign", likelihood="moderate", evidence_for=meningioma,
                        next_tests=["Brain MRI follow-up"], rationale="Typical extra-axial appearance."),
        ]
    elif nodule:
        d = [
            DxCandidate(rank=1, dx="Benign lung nodule (granuloma or scar)", likelihood="high", evidence_for=nodule,
                        next_tests=["Follow-up chest CT per guideline"], rationale="Small incidental nodule."),
            DxCandidate(rank=2, dx="Early lung cancer", likelihood="low", evidence_for=nodule,
                        next_tests=["Follow-up chest CT per guideline"], rationale="Cannot be excluded at one time point."),
        ]
    else:
        conf = [f for f in case.findings if "report_extraction" in f.sources]
        if not conf:
            return ReasonerOut(insufficient_info=True)
        d = [DxCandidate(rank=i + 1, dx=f"{f.finding.capitalize()}, clinical correlation", likelihood="moderate",
                         evidence_for=[f.finding_id], next_tests=["Clinician review"], rationale="Single finding.")
             for i, f in enumerate(conf[:4])]

    return ReasonerOut(differential=d)


def inject_fault(case: Case, rt: Runtime) -> None:
    """OpenHospital fault injection (any backend: mock, LLM or Deep Agent): on the first attempt, plant a
    citation to a finding that doesn't exist. The Verifier must catch it before the doctor sees it."""
    if rt.config.get("pipeline", {}).get("inject_reasoner_fault") and case.verifier.attempts == 0 \
            and case.differential:
        case.differential[0].evidence_for = case.differential[0].evidence_for + ["F-999"]
        case.log("openhospital", "fault injected: F-999 cited on first attempt", rt.clock.iso())


def run(case: Case, rt: Runtime) -> Case:
    if maybe(rt.llm):
        floor = rt.config["pipeline"]["confidence_floor"]
        user = timeline(case)
        if case.verifier.status == "fail" and case.verifier.notes:
            user += "\n\nA reviewer rejected your previous answer. Fix these problems:\n- " + "\n- ".join(case.verifier.notes)
        out = rt.llm.structured("reasoner", SYSTEM.format(floor=floor), user, ReasonerOut)
    else:
        out = _mock_reason(case, rt)
    case.differential = tidy_differential(sorted(out.differential, key=lambda x: x.rank)[:5])
    case.insufficient_info = out.insufficient_info
    inject_fault(case, rt)
    case.state = CaseState.REASONED
    top = case.differential[0].dx if case.differential else "insufficient information"
    case.log("reasoner", f"differential written (top: {top}); attempt {case.verifier.attempts + 1}", rt.clock.iso())
    return case
