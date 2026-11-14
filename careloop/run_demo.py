"""End-to-end demo in the terminal (mock mode: no keys, GPU or Photon needed).

    python run_demo.py            # all scenarios
    python run_demo.py --only P-001

Walks the three stage scenarios plus the 'unrelated findings' case:
  P-003 critical pneumothorax  -> fast path to doctor
  P-002 incidental nodule      -> guideline follow-up, routine texts, closes when the CT arrives
  P-001 lung mass + brain      -> cross-scan reasoning, serious tier, missed appt -> escalation
  P-004 two unrelated findings -> reasoner must NOT link them
"""
from __future__ import annotations

import argparse
import json

from careloop import service
from careloop.runtime import ROOT, default_runtime
from careloop.schema import Patient, Scan

DATA = json.loads((ROOT / "data" / "demo_cases.json").read_text())


def banner(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def show(case) -> None:
    print(f"state={case.state.value} urgency={case.urgency}")
    for f in case.findings:
        print(f"  {f.finding_id} {f.finding:<28} {f.location:<20} conf={f.confidence} src={'+'.join(f.sources)}"
              + (f" linked_to={f.linked_to}" if f.linked_to else ""))
    for d in case.differential:
        print(f"  #{d.rank} {d.dx} [{d.likelihood}] cites {d.evidence_for}")
    print(f"  verifier: {case.verifier.status} after {case.verifier.attempts} attempt(s) {case.verifier.notes or ''}")
    for fu in case.followups:
        print(f"  {fu.id} {fu.action:<34} due {fu.due} status={fu.status} appt={fu.appointment}\n       rule: {fu.rule}")


def scans_for(pid):
    return [(Patient(**DATA["patients"][s["patient_id"]]), Scan(**s["scan"])) for s in DATA["scans"]
            if s["patient_id"] == pid]


def main(only=None):
    rt = default_runtime()
    rt.store.clear()
    rt.clock.set("2026-11-14T09:00")

    if only in (None, "P-003"):
        banner("P-003 | Critical: large pneumothorax -> fast path (skips reasoning)")
        for p, s in scans_for("P-003"):
            show(service.submit_scan(rt, p, s))
        case, _ = service.doctor_reply(rt, "P-003", "CALL")
        print("doctor replied CALL ->", case.state.value)

    if only in (None, "P-002"):
        banner("P-002 | Incidental 7 mm nodule on a kidney-stone CT -> guideline follow-up")
        for p, s in scans_for("P-002"):
            show(service.submit_scan(rt, p, s))
        case, _ = service.doctor_reply(rt, "P-002", "APPROVE")
        show(case)
        service.patient_reply(rt, "P-002", "OK")
        for fu in rt.store.get("P-002").followups:            # December labs/ultrasound come back normal
            if fu.status == "booked" and "chest ct" not in fu.action.lower():
                service.record_result(rt, "P-002", fu.id, "normal")
        banner("P-002 | ...6 months later the follow-up CT arrives: FU-1 closes, nodule stable -> next interval")
        rt.clock.set("2027-05-10T09:00")
        p = Patient(**DATA["patients"]["P-002"])
        follow = Scan(scan_id="CT-P002-2", modality="CT", body_part="CHEST", date="2027-05-10",
                      report_text="CT chest without contrast. The 7 mm nodule in the right lower lobe is unchanged.")
        case = service.submit_scan(rt, p, follow)
        case = service.advance_clock(rt, minutes=1)[[c.patient.patient_id for c in rt.store.list()].index("P-002")]
        show(case)
        rt.clock.set("2026-11-14T09:00")

    if only in (None, "P-001"):
        banner("P-001 | Chest X-ray: right hilar mass (urgent)")
        s1, s2 = scans_for("P-001")
        show(service.submit_scan(rt, *s1))
        service.doctor_reply(rt, "P-001", "APPROVE")
        banner("P-001 | Two weeks later, brain MRI: 3 ring-enhancing lesions -> cross-scan reasoning")
        show(service.submit_scan(rt, *s2))
        banner("P-001 | Doctor doesn't answer -> reminder, then escalation (fast-forward)")
        service.advance_clock(rt, minutes=16)
        service.advance_clock(rt, minutes=20)
        case, _ = service.doctor_reply(rt, "P-001", "APPROVE")
        show(case)
        banner("P-001 | Chest CT happens (closes FU-1); PET-CT is missed -> re-book offer -> patient reschedules")
        def find(pred):
            return next((f for f in case.followups if f.appointment and pred(f.action.lower())), None)
        ct = find(lambda a: ("chest ct" in a or "ct chest" in a) and "pet" not in a)
        pet = find(lambda a: "pet" in a)
        if not ct or not pet:
            print("  (model chose a different plan; skipping the missed-appointment scene)")
            ct = pet = None
    if only in (None, "P-001") and ct and pet:
        rt.clock.set(ct.appointment)
        service.submit_scan(rt, Patient(**DATA["patients"]["P-001"]),
                            Scan(scan_id="CT-P001-1", modality="CT", body_part="CHEST", date=ct.appointment[:10],
                                 report_text="CT chest with contrast. The right hilar mass measures 3.4 cm. No pleural effusion."))
        service.doctor_reply(rt, "P-001", "APPROVE")
        rt.clock.set(pet.appointment)
        service.advance_clock(rt, hours=2, minutes=5)       # PET-CT missed-check fires
        service.patient_reply(rt, "P-001", "RESCHEDULE")
        show(rt.store.get("P-001"))
    if only in (None, "P-001"):
        banner("P-001 | Patient asks a clinical question -> handed to the care team, never answered by the agent")
        service.patient_reply(rt, "P-001", "Is it cancer?")

    if only in (None, "P-004"):
        banner("P-004 | Two unrelated findings -> must NOT be linked")
        for p, s in scans_for("P-004"):
            case = service.submit_scan(rt, p, s)
        show(case)

    banner("Audit log sample (P-001)")
    c = rt.store.get("P-001")
    if c:
        for e in c.audit_log[-12:]:
            print(f"  {e.at} [{e.agent}] {e.action}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    main(ap.parse_args().only)
