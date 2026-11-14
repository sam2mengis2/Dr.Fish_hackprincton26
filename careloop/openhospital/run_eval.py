"""Run the OpenHospital benchmark for one configuration and append a leaderboard row.

    python -m openhospital.run_eval --name baseline
    python -m openhospital.run_eval --name no-verifier --set pipeline.verifier_enabled=false
    python -m openhospital.run_eval --name sonnet --set llm.provider=anthropic --runs 3
    python -m openhospital.run_eval --split holdout --name final

One config edit = one experiment. Freeze tasks + graders before comparing configs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import statistics
from datetime import datetime
from pathlib import Path

import yaml

from careloop.runtime import ROOT, load_config

from .env import OpenHospitalEnv
from .tasks import TASKS


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for s in sets:
        key, val = s.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--set", action="append", default=[], help="override, e.g. pipeline.verifier_enabled=false")
    ap.add_argument("--runs", type=int, default=1, help="repeat each task (LLMs vary run to run)")
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    ap.add_argument("--name", default="run")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg = apply_overrides(load_config(a.config), a.set)
    env = OpenHospitalEnv(cfg)
    tasks = [t for t in TASKS.values() if a.split == "all" or t.split == a.split]
    per_task, secs, calls = {}, [], 0
    for t in tasks:
        scores = []
        for _ in range(a.runs):
            env.reset(t.id)
            r = env.run()
            env.close()
            scores.append(r.score)
            secs.append(r.seconds)
            calls += r.llm_calls
            if not a.quiet:
                failed = [k for k, v in r.checks.items() if not v]
                print(f"{t.id:<26} [{t.split:<7}] score={r.score:.2f}  "
                      + (f"FAILED: {failed}" if failed else "all checks pass")
                      + (f"  ERROR: {r.error}" if r.error else ""))
        per_task[t.id] = scores

    means = [statistics.mean(s) for s in per_task.values()]
    overall = statistics.mean(means) if means else 0.0
    spread = (min(min(s) for s in per_task.values()), max(max(s) for s in per_task.values())) if per_task else (0, 0)
    print(f"\n== {a.name}: overall {overall:.3f}  (task-score range {spread[0]:.2f}-{spread[1]:.2f}, "
          f"{len(tasks)} tasks x {a.runs} runs, {sum(secs):.1f}s, {calls} LLM calls)")

    out = ROOT / "runs" / "leaderboard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["when", "name", "split", "runs", "overall", "llm_provider", "reasoner_model", "verifier",
                        "seconds", "llm_calls", "overrides"] + list(TASKS))
        w.writerow([datetime.now().isoformat(timespec="seconds"), a.name, a.split, a.runs, f"{overall:.3f}",
                    cfg["llm"]["provider"], cfg["llm"].get("reasoner_model"), cfg["pipeline"]["verifier_enabled"],
                    f"{sum(secs):.1f}", calls, ";".join(a.set)]
                   + [f"{statistics.mean(per_task[t]):.2f}" if t in per_task else "" for t in TASKS])
    print(f"leaderboard row appended -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
