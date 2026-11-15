"""Runtime plumbing shared by all agents: config, simulated clock, case store, messenger.

Swap implementations here without touching agent code:
  - Store: JSON file now; Postgres later (same get/save/list API).
  - Messenger: Console now; Photon behind the same send() interface.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import yaml

from .schema import Case

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- config
def load_config(path: Optional[str] = None) -> dict:
    p = Path(path or os.environ.get("CARELOOP_CONFIG", ROOT / "config.yaml"))
    with open(p) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------- clock
class SimClock:
    """Simulated clock so a 30-day follow-up can run in seconds on stage."""

    def __init__(self, start: str = "2026-11-14T09:00"):
        self._now = datetime.fromisoformat(start)

    def now(self) -> datetime:
        return self._now

    def iso(self) -> str:
        return self._now.isoformat(timespec="minutes")

    def advance(self, **kwargs) -> datetime:
        self._now += timedelta(**kwargs)
        return self._now

    def set(self, iso: str) -> None:
        self._now = datetime.fromisoformat(iso)


# ---------------------------------------------------------------- store
class CaseStore:
    """One JSON file per patient. Same API you'd keep for a Postgres JSONB table."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, patient_id: str) -> Path:
        return self.dir / f"{patient_id}.json"

    def get(self, patient_id: str) -> Optional[Case]:
        p = self._path(patient_id)
        if not p.exists():
            return None
        return Case.model_validate_json(p.read_text())

    def save(self, case: Case) -> None:
        with self._lock:
            self._path(case.patient.patient_id).write_text(case.model_dump_json(indent=2))

    def list(self) -> list[Case]:
        return [Case.model_validate_json(p.read_text()) for p in sorted(self.dir.glob("*.json"))]

    def clear(self) -> None:
        for p in self.dir.glob("*.json"):
            p.unlink()


# ---------------------------------------------------------------- messaging
class Messenger:
    """send(to_role, patient_id, text). Inbound replies arrive via the API -> clinician_comms.handle_reply."""

    def send(self, to: str, patient_id: str, text: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ConsoleMessenger(Messenger):
    def __init__(self, sink: Optional[Callable[[str], None]] = None, quiet: bool = False):
        self.sink = sink or print
        self.quiet = quiet
        self.outbox: list[dict] = []

    def send(self, to: str, patient_id: str, text: str) -> None:
        self.outbox.append({"to": to, "patient_id": patient_id, "text": text})
        if not self.quiet:
            self.sink(f"\n[TEXT -> {to} | {patient_id}]\n{text}\n")




# ---------------------------------------------------------------- context
class Runtime:
    """Everything an agent needs, passed explicitly (easy to swap in tests and evals)."""

    def __init__(self, config: dict, store: CaseStore, clock: SimClock, messenger: Messenger, llm=None):
        self.config = config
        self.store = store
        self.clock = clock
        self.messenger = messenger
        if llm is None:
            from .llm import make_llm
            llm = make_llm(config)
        self.llm = llm


def default_runtime(config_path: Optional[str] = None, quiet: bool = False) -> Runtime:
    cfg = load_config(config_path)
    store = CaseStore(ROOT / cfg.get("store_dir", "runs/cases"))
    clock = SimClock(cfg.get("clock_start", "2026-11-14T09:00"))
    if cfg.get("messenger") == "photon":
        from .photon_messenger import PhotonMessenger

        messenger: Messenger = PhotonMessenger(ROOT / "runs" / "photon_registry.json",
                                               console_fallback=ConsoleMessenger(quiet=quiet))
    else:
        messenger = ConsoleMessenger(quiet=quiet)
    return Runtime(cfg, store, clock, messenger)


def dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
