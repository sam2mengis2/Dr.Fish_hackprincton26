"""FastAPI backend. Run:  uvicorn api:app --reload --port 8000

Endpoints
  POST /scans                {patient: {...}, scan: {...}}     -> runs the loop until the doctor is texted
  POST /scans/acquire        {patient, scan} images taken, report not signed yet -> reading queue
  POST /reports/sign         {patient_id, scan_id, report_text} radiologist signs -> loop runs
  GET  /reading-queue        unsigned scans, stat first, then urgent, then routine
  GET  /waitlist             every booked scan, most urgent clinical due date first (+ open offers)
  POST /cancel               {patient_id, followup_id?} frees the slot -> offered to whoever needs it most
  POST /demo/load/{pid}      advances that patient's demo story one step (data/demo_cases.json)
  GET  /cases                worklist, sorted by urgency
  GET  /cases/{pid}          the full shared case record
  POST /reply/doctor         {patient_id, text}   (also the target for Photon's inbound webhook)
  POST /reply/patient        {patient_id, text}
  POST /clock/advance        {minutes?, hours?, days?}  -> watchdog fires due timers
  POST /reset                clears all cases (demo)
  GET  /messages             the console outbox (what would have been texted)
"""
from __future__ import annotations

import json
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from careloop import service
from careloop.agents.triage import worklist_key
from careloop.runtime import ROOT, default_runtime
from careloop.schema import Patient, Scan

rt = default_runtime()
app = FastAPI(title="CareLoop")
LOCK = threading.RLock()  # phone replies (Photon thread) and dashboard clicks must not edit a case at once


@app.on_event("startup")
def start_photon():
    from careloop.photon_messenger import PhotonMessenger

    if isinstance(rt.messenger, PhotonMessenger):
        def on_doctor(pid, text):
            with LOCK:
                service.doctor_reply(rt, pid, text)

        def on_patient(pid, text):
            with LOCK:
                service.patient_reply(rt, pid, text)

        rt.messenger.start_polling(on_doctor, on_patient)


@app.get("/photon/status")
def photon_status():
    from careloop.photon_messenger import PhotonMessenger

    m = rt.messenger
    if not isinstance(m, PhotonMessenger):
        return {"messenger": "console"}
    return {"messenger": "photon", "agent_number": m.agent_handle, "solo_mode": m.solo(),
            "registered": {k: v.get("handle") for k, v in m.registry.items()}}
DEMO = json.loads((ROOT / "data" / "demo_cases.json").read_text())


class ScanIn(BaseModel):
    patient: Patient
    scan: Scan


class SignIn(BaseModel):
    patient_id: str
    scan_id: str
    report_text: str


class ReplyIn(BaseModel):
    patient_id: str
    text: str


class ClockIn(BaseModel):
    minutes: int = 0
    hours: int = 0
    days: int = 0


@app.post("/scans")
def post_scan(body: ScanIn):
    try:
        with LOCK:
            return service.submit_scan(rt, body.patient, body.scan)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/demo/load/{pid}")
def demo_load(pid: str):
    """One click = one step of that patient's story: outside priors are imported first; a scan marked
    report_delay is first acquired (report pending, reading queue) and signed on the next click; a
    follow-up scan arrives on the day of its booked appointment."""
    with LOCK:
        patient = Patient(**DEMO["patients"][pid])
        case = rt.store.get(pid)
        have = {s.scan_id: s for s in (case.scans if case else [])}
        for o in DEMO.get("outside_records", []):
            if o["patient_id"] == pid and o["scan"]["scan_id"] not in have:
                service.import_outside_record(rt, patient, Scan(**o["scan"]))
        for e in (e for e in DEMO["scans"] if e["patient_id"] == pid):
            sid = e["scan"]["scan_id"]
            if sid in have and have[sid].report_status == "pending":
                return service.sign_report(rt, pid, sid, e["scan"]["report_text"])
            if sid in have:
                continue
            scan = Scan(**e["scan"])
            if e.get("followup"):
                case = rt.store.get(pid)
                fu = min((f for f in case.followups if f.appointment and f.status == "booked"),
                         key=lambda f: f.appointment, default=None) if case else None
                if fu is None:
                    raise HTTPException(409, "approve the plan first so the follow-up gets booked")
                appt = fu.appointment
                if rt.clock.iso() < appt:
                    rt.clock.set(appt)
                    for c in rt.store.list():
                        rt.store.save(watchdog_tick(c))
                scan.date, scan.site = appt[:10], fu.site or scan.site
            if e.get("report_delay"):
                return service.acquire_scan(rt, patient, scan)
            return service.submit_scan(rt, patient, scan)
    raise HTTPException(404, f"no more demo steps for {pid}")


@app.post("/demo/busy")
def demo_busy():
    """Turn on the demo scanner backlogs and seed patients already booked (idempotent)."""
    with LOCK:
        added = service.seed_busy_scanners(rt, DEMO)
    return {"backlog_days": rt.config["scheduling"]["backlog_days"], "added": added}


@app.post("/demo/cancel")
def demo_cancel():
    """A patient holding an early slot cancels -> CareLoop offers it to whoever needs it most."""
    who = next((w["patient"]["patient_id"] for w in (DEMO.get("busy_scanners") or {}).get("waiting_patients", [])
                if w.get("cancels")), None)
    if not who or not rt.store.get(who):
        raise HTTPException(409, "click 'Simulate morning' first")
    with LOCK:
        service.cancel_appointment(rt, who, reason="patient got the scan elsewhere")
    return service.waitlist(rt)


class CancelIn(BaseModel):
    patient_id: str
    followup_id: str | None = None
    reason: str = ""


@app.post("/cancel")
def cancel(body: CancelIn):
    with LOCK:
        service.cancel_appointment(rt, body.patient_id, body.followup_id, body.reason)
    return service.waitlist(rt)


@app.get("/metrics")
def metrics():
    return service.metrics(rt)


@app.get("/waitlist")
def waitlist():
    return service.waitlist(rt)


def watchdog_tick(case):
    from careloop.agents import watchdog

    return watchdog.tick(case, rt)


@app.get("/cases")
def cases():
    return [{"patient_id": c.patient.patient_id, "name": c.patient.name, "state": c.state, "urgency": c.urgency,
             "top": c.differential[0].dx if c.differential else None,
             "open_followups": sum(f.status != "completed" for f in c.followups)}
            for c in sorted(rt.store.list(), key=worklist_key)]


@app.get("/cases/{pid}")
def case(pid: str):
    c = rt.store.get(pid)
    if not c:
        raise HTTPException(404)
    return c


@app.post("/scans/acquire")
def acquire(body: ScanIn):
    try:
        with LOCK:
            return service.acquire_scan(rt, body.patient, body.scan)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/reports/sign")
def sign(body: SignIn):
    try:
        with LOCK:
            return service.sign_report(rt, body.patient_id, body.scan_id, body.report_text)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/reading-queue")
def reading_queue():
    return service.reading_queue(rt)


@app.post("/reply/doctor")
def reply_doctor(body: ReplyIn):
    with LOCK:
        c, outcome = service.doctor_reply(rt, body.patient_id, body.text)
    return {"outcome": outcome, "state": c.state}


@app.post("/reply/patient")
def reply_patient(body: ReplyIn):
    with LOCK:
        c, outcome = service.patient_reply(rt, body.patient_id, body.text)
    return {"outcome": outcome, "state": c.state}


class ResultIn(BaseModel):
    patient_id: str
    followup_id: str
    note: str = ""


@app.post("/results")
def post_result(body: ResultIn):
    """Lab / consult result arrived -> closes that follow-up (imaging closes via POST /scans)."""
    try:
        with LOCK:
            c = service.record_result(rt, body.patient_id, body.followup_id, body.note)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"state": c.state, "followups": [(f.id, f.status) for f in c.followups]}


@app.post("/clock/advance")
def advance(body: ClockIn):
    with LOCK:
        service.advance_clock(rt, minutes=body.minutes, hours=body.hours, days=body.days)
    return {"now": rt.clock.iso()}


@app.get("/clock")
def clock():
    return {"now": rt.clock.iso()}


@app.post("/reset")
def reset():
    rt.store.clear()
    rt.config.setdefault("scheduling", {})["backlog_days"] = {}
    rt.clock.set(rt.config.get("clock_start", "2026-11-14T09:00"))
    for attr in ("calendar", "calendars"):
        if hasattr(rt, attr):
            delattr(rt, attr)
    if hasattr(rt.messenger, "outbox"):
        rt.messenger.outbox.clear()
    return {"ok": True}


@app.get("/messages")
def messages():
    return getattr(rt.messenger, "outbox", [])
