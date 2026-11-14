"""FastAPI backend. Run:  uvicorn api:app --reload --port 8000

Endpoints
  POST /scans                {patient: {...}, scan: {...}}     -> runs the loop until the doctor is texted
  POST /demo/load/{pid}      loads that patient's next demo scan from data/demo_cases.json
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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from careloop import service
from careloop.agents.triage import worklist_key
from careloop.runtime import ROOT, default_runtime
from careloop.schema import Patient, Scan

rt = default_runtime()
app = FastAPI(title="CareLoop")
DEMO = json.loads((ROOT / "data" / "demo_cases.json").read_text())


class ScanIn(BaseModel):
    patient: Patient
    scan: Scan


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
        return service.submit_scan(rt, body.patient, body.scan)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/demo/load/{pid}")
def demo_load(pid: str):
    done = {s.scan_id for s in (rt.store.get(pid).scans if rt.store.get(pid) else [])}
    nxt = next((s for s in DEMO["scans"] if s["patient_id"] == pid and s["scan"]["scan_id"] not in done), None)
    if not nxt:
        raise HTTPException(404, f"no more demo scans for {pid}")
    return service.submit_scan(rt, Patient(**DEMO["patients"][pid]), Scan(**nxt["scan"]))


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


@app.post("/reply/doctor")
def reply_doctor(body: ReplyIn):
    c, outcome = service.doctor_reply(rt, body.patient_id, body.text)
    return {"outcome": outcome, "state": c.state}


@app.post("/reply/patient")
def reply_patient(body: ReplyIn):
    c, outcome = service.patient_reply(rt, body.patient_id, body.text)
    return {"outcome": outcome, "state": c.state}


@app.post("/clock/advance")
def advance(body: ClockIn):
    service.advance_clock(rt, minutes=body.minutes, hours=body.hours, days=body.days)
    return {"now": rt.clock.iso()}


@app.get("/clock")
def clock():
    return {"now": rt.clock.iso()}


@app.post("/reset")
def reset():
    rt.store.clear()
    rt.clock.set(rt.config.get("clock_start", "2026-11-14T09:00"))
    if hasattr(rt, "calendar"):
        del rt.calendar
    if hasattr(rt.messenger, "outbox"):
        rt.messenger.outbox.clear()
    return {"ok": True}


@app.get("/messages")
def messages():
    return getattr(rt.messenger, "outbox", [])
