"""Photon 2.0 (Beta) messaging for CareLoop: doctor approvals and patient updates over iMessage.

Uses Photon's official Python client (pip install photonhq-api==0.5.0, Python 3.11+).
  send     -> photon.projects.send_message   (POST /v1/projects/{id}/messages)
  receive  -> photon.projects.list_messages  (GET  /v1/projects/{id}/messages), polled every 2 s.
              Polling means no public URL / tunnel is needed for the demo.

Photon rule: a shared iMessage line can't send the first message. So each phone REGISTERS by
texting the CareLoop number once:
    DOCTOR            -> this phone is the doctor's phone
    PATIENT           -> this phone is the patient's phone
    DOCTOR PATIENT    -> one phone plays both roles (solo demo mode)
After that the agent can message that phone. In solo mode outgoing texts are labelled
"[To Dr. Lee]" / "[To patient]" and replies are routed by keyword:
    APPROVE / EDIT ... / REJECT ... / CALL   -> doctor flow
    anything else (OK, RESCHEDULE, questions) -> patient flow
Optionally start a reply with a patient ID to target a case: "P-002 APPROVE".

Environment (never put these in config.yaml or git):
    PHOTON_API_KEY      server-side key with project:read + project:write
    PHOTON_PROJECT_ID   your Beta project id
    PHOTON_AGENT_HANDLE the shared-line number assigned to your project (E.164, e.g. +15551234567)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from .runtime import Messenger

DOCTOR_WORDS = re.compile(r"^\s*(P-\d{3}\s+)?(APPROVE|APPROVED|YES|EDIT|REJECT|CALL)\b", re.I)
REGISTER = re.compile(r"^\s*(DOCTOR|PATIENT)(\s+(DOCTOR|PATIENT))?\s*$", re.I)
PID = re.compile(r"^\s*(P-\d{3})\s+(.*)$", re.I | re.S)
ROLE_OF = {"doctor": "doctor", "covering_doctor": "doctor", "nurse": "doctor", "radiologist": "doctor",
           "patient": "patient"}
LABEL = {"doctor": "[To ordering doctor]", "covering_doctor": "[To covering doctor]", "nurse": "[To nurse navigator]",
         "radiologist": "[To radiology]",
         "patient": "[To patient]"}


class PhotonMessenger(Messenger):
    def __init__(self, registry_path: str | Path, console_fallback: Optional[Messenger] = None,
                 client=None, poll_s: float = 2.0):
        self.key = os.environ.get("PHOTON_API_KEY", "").strip()
        self.project = os.environ.get("PHOTON_PROJECT_ID", "").strip()
        self.agent_handle = os.environ.get("PHOTON_AGENT_HANDLE", "").strip()
        missing = [n for n, v in (("PHOTON_API_KEY", self.key), ("PHOTON_PROJECT_ID", self.project),
                                  ("PHOTON_AGENT_HANDLE", self.agent_handle)) if not v]
        if missing and client is None:
            raise RuntimeError(f"Photon not configured; set {', '.join(missing)} (see careloop/photon_messenger.py)")
        self.registry_path = Path(registry_path)
        self.registry: dict = self._load()          # {"doctor": {"spaceId","senderId","handle"}, "patient": {...}}
        self.fallback = console_fallback
        self.poll_s = poll_s
        self._client = client
        self._seen: set[str] = set()
        self._last_pid: dict[str, str] = {}         # role -> patient_id of the last message sent to that role
        self._stop = threading.Event()
        self.outbox: list[dict] = []

    # ------------------------------------------------------------------ client
    def _photon(self):
        if self._client is None:
            from photon_api import Photon

            self._client = Photon(headers={"Authorization": f"Bearer {self.key}"})
        return self._client

    # ------------------------------------------------------------------ registry
    def _load(self) -> dict:
        try:
            return json.loads(self.registry_path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.registry, indent=2))

    def solo(self) -> bool:
        d, p = self.registry.get("doctor"), self.registry.get("patient")
        return bool(d and p and d["spaceId"] == p["spaceId"])

    # ------------------------------------------------------------------ send
    def send(self, to: str, patient_id: str, text: str) -> None:
        role = ROLE_OF.get(to, "doctor")
        if to in ("doctor", "patient"):  # FYI alerts (radiology, nurse) shouldn't redirect the next APPROVE
            self._last_pid[role] = patient_id
        self.outbox.append({"to": to, "patient_id": patient_id, "text": text})
        target = self.registry.get(role)
        if not target:
            msg = f"(Photon: no phone registered as {role.upper()} yet; text {role.upper()} to {self.agent_handle})"
            if self.fallback:
                self.fallback.send(to, patient_id, f"{msg}\n{text}")
            else:
                print(msg)
            return
        body = text
        if self.solo() or to in ("covering_doctor", "nurse"):
            label = LABEL.get(to, "")
            if patient_id and to in ("doctor", "patient"):
                label = label[:-1] + f" {patient_id}]"
            body = f"{label} {text}".strip()
        self._send_raw(target["spaceId"], target["senderId"], body, key=f"careloop:{uuid.uuid4()}")

    def _send_raw(self, space_id: str, sender_id: str, text: str, key: str, parent: Optional[str] = None) -> None:
        from photon_api.generated.models import SendMessageRequest, SendTextContent
        from photon_api.rpc_generated import SendMessageHeader, SendMessageInput, SendMessagePath

        body = dict(spaceId=space_id, senderId=sender_id,
                    content=SendTextContent(type="text", format="plain", text=text[:1500]))
        if parent:
            body["parentMessageId"] = parent
        try:
            self._photon().projects.send_message(SendMessageInput(
                path=SendMessagePath(id=self.project),
                headers=SendMessageHeader.model_validate({"idempotency-key": key}),
                body=SendMessageRequest(**body)))
        except Exception as e:  # never let messaging crash the clinical loop
            print(f"[photon] send failed ({type(e).__name__}: {e}); message kept in CareLoop's record")
            if self.fallback:
                self.fallback.send("doctor", "", f"(Photon send failed) {text}")

    # ------------------------------------------------------------------ receive
    def handle_inbound(self, space_id: str, agent_user_id: str, from_handle: str, text: str, msg_id: str,
                       on_doctor: Callable[[str, str], None], on_patient: Callable[[str, str], None]) -> str:
        """Route one inbound text. Returns what it did (for logs/tests)."""
        m = REGISTER.match(text)
        if m:
            roles = {r.lower() for r in (m.group(1), m.group(3)) if r}
            for r in roles:
                self.registry[r] = {"spaceId": space_id, "senderId": agent_user_id, "handle": from_handle}
            self._save()
            who = " and ".join(sorted(roles))
            self._send_raw(space_id, agent_user_id,
                           f"CareLoop: this phone is now registered as {who}. You'll get "
                           f"{'approvals and updates' if len(roles) > 1 else ('approval requests' if 'doctor' in roles else 'appointment updates')} here.",
                           key=f"careloop-reg:{msg_id}", parent=msg_id)
            return f"registered:{who}"

        roles_here = [r for r, v in self.registry.items() if v.get("spaceId") == space_id]
        if not roles_here:
            self._send_raw(space_id, agent_user_id, "CareLoop: reply DOCTOR or PATIENT to register this phone.",
                           key=f"careloop-help:{msg_id}", parent=msg_id)
            return "unregistered"
        if len(roles_here) == 1:
            role = roles_here[0]
        else:
            role = "doctor" if DOCTOR_WORDS.match(text) else "patient"

        pid = self._last_pid.get(role)
        pm = PID.match(text)
        if pm:
            pid, text = pm.group(1).upper(), pm.group(2)
        if not pid:
            self._send_raw(space_id, agent_user_id, "CareLoop: which patient? Start your reply with the ID, e.g. 'P-002 APPROVE'.",
                           key=f"careloop-pid:{msg_id}", parent=msg_id)
            return "no-patient"
        (on_doctor if role == "doctor" else on_patient)(pid, text)
        return f"{role}:{pid}"

    def start_polling(self, on_doctor: Callable[[str, str], None], on_patient: Callable[[str, str], None]) -> threading.Thread:
        t = threading.Thread(target=self._poll_loop, args=(on_doctor, on_patient), daemon=True, name="photon-poll")
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()

    def _poll_loop(self, on_doctor, on_patient) -> None:
        from photon_api.generated.models import TextContent, UserSnapshot
        from photon_api.rpc_generated import ListMessagesInput

        since = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        print(f"[photon] polling for texts to {self.agent_handle} (registered: {list(self.registry) or 'none'})")
        while not self._stop.is_set():
            try:
                query = {"createdSince": since, "direction": "received", "order": "asc", "pageSize": 100,
                         "platform": "imessage"}
                while True:
                    page = self._photon().projects.list_messages(
                        ListMessagesInput.model_validate({"path": {"id": self.project}, "query": query}))
                    for item in page.items:
                        m = item.root.root
                        if m.id in self._seen or m.status != "received" or not isinstance(m.receivedBy, UserSnapshot):
                            continue
                        self._seen.add(m.id)
                        sender, agent, space = m.sentBy.root, m.receivedBy.root, m.space.root
                        if sender.kind != "human" or agent.kind != "agent" or not isinstance(m.content, TextContent):
                            continue
                        try:
                            what = self.handle_inbound(space.id, agent.id, sender.handle, m.content.text, m.id,
                                                       on_doctor, on_patient)
                            print(f"[photon] {sender.handle}: {m.content.text!r} -> {what}")
                        except Exception as e:
                            print(f"[photon] handling failed: {type(e).__name__}: {e}")
                    token = page.nextPageToken
                    if not isinstance(token, str):
                        break
                    query["pageToken"] = token
            except Exception as e:
                print(f"[photon] poll error ({type(e).__name__}: {e}); retrying")
            self._stop.wait(self.poll_s)
