"""OpenHospital: a simulated hospital for evaluating CareLoop configurations.

Each episode = one task scenario run against a FRESH, isolated runtime (own store, clock,
calendar, quiet messenger). Simulated doctor and patient act from a script. Graders are
plain code and return a score in [0, 1] with per-check details.

Gymnasium-style surface:
    env = OpenHospitalEnv(config)
    obs = env.reset(task_id)          # builds the world
    result = env.run()                # runs the scripted episode with CareLoop as the agent
    result.score, result.checks
"""
from __future__ import annotations

import copy
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from careloop import service
from careloop.runtime import CaseStore, ConsoleMessenger, Runtime, SimClock

from .tasks import TASKS, Task


@dataclass
class EpisodeResult:
    task_id: str
    score: float
    checks: dict[str, bool]
    seconds: float
    llm_calls: int
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class OpenHospitalEnv:
    def __init__(self, config: dict):
        self.base_config = config
        self.task: Task | None = None
        self.rt: Runtime | None = None
        self._tmp: Path | None = None

    def reset(self, task_id: str) -> dict:
        self.close()
        self.task = TASKS[task_id]
        cfg = copy.deepcopy(self.base_config)
        for k, v in self.task.config_overrides.items():   # e.g. fault injection for the Verifier task
            cfg.setdefault(k.split(".")[0], {})[k.split(".")[1]] = v
        self._tmp = Path(tempfile.mkdtemp(prefix="openhospital_"))
        self.rt = Runtime(cfg, CaseStore(self._tmp), SimClock(self.task.clock_start), ConsoleMessenger(quiet=True))
        return {"task": task_id, "description": self.task.description}

    def run(self) -> EpisodeResult:
        assert self.task and self.rt
        t0 = time.time()
        try:
            ctx = self.task.script(self.rt, service)
            checks = self.task.grade(self.rt, ctx)
            err = None
        except Exception as e:  # a crash scores 0 but doesn't stop the benchmark
            checks, err = {"ran_without_error": False}, f"{type(e).__name__}: {e}"
        score = sum(checks.values()) / max(1, len(checks))
        return EpisodeResult(self.task.id, round(score, 3), checks, round(time.time() - t0, 2),
                             getattr(self.rt.llm, "calls", 0), err)

    def close(self) -> None:
        if self._tmp and self._tmp.exists():
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = None
