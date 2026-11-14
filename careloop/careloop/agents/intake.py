"""Agent 1 - Intake & Router (plain code).

Creates or updates the patient's case, routes the scan to the right workers,
and closes any follow-up item this scan fulfils (closing the loop).
"""
from __future__ import annotations

from ..runtime import Runtime
from ..schema import Case, CaseState, Patient, Scan

ROUTES = {
    ("CR", "CHEST"): ["cxr_model"],
    ("DX", "CHEST"): ["cxr_model"],
    ("MR", "BRAIN"): ["brain_mri_model"],
    ("MR", "HEAD"): ["brain_mri_model"],
}

# Which follow-up actions a new scan can satisfy: (modality, body_part) -> keywords in the action
FULFILS = {
    ("CT", "CHEST"): ["chest ct", "ct chest"],
    ("PT", "WHOLEBODY"): ["pet-ct", "pet/ct"],
    ("PT", "CHEST"): ["pet-ct", "pet/ct"],
    ("MR", "BRAIN"): ["brain mri", "mri brain"],
    ("CR", "CHEST"): ["chest x-ray", "repeat cxr"],
}


def route(scan: Scan) -> list[str]:
    models = ROUTES.get((scan.modality.upper(), scan.body_part.upper()), ["fallback_model"])
    return models + (["report_extraction"] if scan.report_text.strip() else [])


def ingest(rt: Runtime, patient: Patient, scan: Scan) -> Case:
    case = rt.store.get(patient.patient_id) or Case(patient=patient)
    if any(s.scan_id == scan.scan_id for s in case.scans):
        raise ValueError(f"scan {scan.scan_id} already ingested")
    scan.route = route(scan)
    case.scans.append(scan)
    case.pending_scan_ids.append(scan.scan_id)
    case.raw_findings = []
    case.verifier.attempts = 0
    case.verifier.status = "pending"
    case.verifier.notes = []
    _complete_followups(case, scan, rt)
    case.state = CaseState.INGESTED
    case.log("intake", f"ingested {scan.scan_id} ({scan.modality}/{scan.body_part}) -> {scan.route}", rt.clock.iso())
    return case


def _complete_followups(case: Case, scan: Scan, rt: Runtime) -> None:
    keys = FULFILS.get((scan.modality.upper(), scan.body_part.upper()), [])
    # One scan completes ONE item: the earliest-due open item it matches.
    matches = sorted((fu for fu in case.followups
                      if fu.status != "completed" and any(k in fu.action.lower() for k in keys)),
                     key=lambda f: f.due)
    for fu in matches[:1]:
        fu.status = "completed"
        fu.completed_by_scan = scan.scan_id
        for t in case.timers:
            if t.ref == fu.id:
                t.done = True
        case.log("intake", f"{fu.id} completed by {scan.scan_id} (loop closed for this item)", rt.clock.iso())
