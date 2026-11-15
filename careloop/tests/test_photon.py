"""Photon messaging, tested offline with a fake Photon client (no keys, no phone)."""
import pytest

pytest.importorskip("photon_api")

from careloop import service
from careloop.photon_messenger import PhotonMessenger
from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock, load_config
from openhospital.tasks import P, S


class FakeProjects:
    def __init__(self):
        self.sent = []

    def send_message(self, inp):
        b = inp.body
        self.sent.append({"space": b.spaceId, "sender": b.senderId, "text": b.content.text})


class FakePhoton:
    def __init__(self):
        self.projects = FakeProjects()


def _setup(tmp_path):
    fake = FakePhoton()
    msgr = PhotonMessenger(tmp_path / "reg.json", console_fallback=ConsoleMessenger(quiet=True), client=fake)
    msgr.project, msgr.agent_handle = "pho_prj_test", "+15550000000"
    cfg = load_config()
    cfg["llm"]["provider"] = "mock"           # tests never call a real model, whatever config.yaml says
    cfg["cv"]["cxr_backend"] = "mock"
    rt = Runtime(cfg, CaseStore(tmp_path / "cases"), SimClock("2026-11-14T09:00"), msgr)
    return rt, msgr, fake


def test_solo_phone_full_loop(tmp_path):
    rt, msgr, fake = _setup(tmp_path)
    doctor = lambda pid, t: service.doctor_reply(rt, pid, t)
    patient = lambda pid, t: service.patient_reply(rt, pid, t)

    # 1. one phone registers as both roles
    assert msgr.handle_inbound("space-1", "agent-1", "+15551234567", "DOCTOR PATIENT", "m1", doctor, patient) \
        == "registered:doctor and patient"
    assert msgr.solo()

    # 2. a scan arrives -> doctor alert goes to that phone, labelled
    service.submit_scan(rt, P("P-002"), S("CT-P002-1"))
    assert fake.projects.sent[-1]["text"].startswith("[To ordering doctor P-002]")
    assert fake.projects.sent[-1]["space"] == "space-1"

    # 3. doctor replies APPROVE from the phone -> patient gets a labelled booking text
    assert msgr.handle_inbound("space-1", "agent-1", "+15551234567", "APPROVE", "m2", doctor, patient) == "doctor:P-002"
    assert rt.store.get("P-002").state.value == "active"
    assert any(s["text"].startswith("[To patient P-002]") and "2027-05" in s["text"] for s in fake.projects.sent)

    # 4. patient replies OK -> routed to the patient flow
    assert msgr.handle_inbound("space-1", "agent-1", "+15551234567", "OK", "m3", doctor, patient) == "patient:P-002"


def test_unregistered_and_explicit_patient_id(tmp_path):
    rt, msgr, fake = _setup(tmp_path)
    seen = []
    assert msgr.handle_inbound("s9", "a9", "+1999", "hello", "x1", lambda *a: None, lambda *a: None) == "unregistered"
    msgr.handle_inbound("s1", "a1", "+1555", "DOCTOR", "x2", lambda *a: None, lambda *a: None)
    out = msgr.handle_inbound("s1", "a1", "+1555", "P-001 APPROVE", "x3", lambda pid, t: seen.append((pid, t)),
                              lambda *a: None)
    assert out == "doctor:P-001" and seen == [("P-001", "APPROVE")]


def test_send_without_registration_falls_back(tmp_path):
    rt, msgr, fake = _setup(tmp_path)
    msgr.send("patient", "P-002", "hi")
    assert fake.projects.sent == []  # nothing sent to Photon; console fallback got it
