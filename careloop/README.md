# CareLoop

From any scan finding to a completed follow-up. Twelve agents in one LangGraph graph read
each scan (computer vision + the radiologist's report), reason across the patient's scans,
get a doctor's approval by text, inform the patient, book follow-ups, and chase every item
until the follow-up scan actually happens.

**Decision support, not diagnosis.** A doctor approves every clinical action. All patients
and reports in `data/` are synthetic.

## Quickstart (mock mode: no API keys, GPU or Photon needed)

```bash
pip install -r requirements.txt
python run_demo.py                     # whole loop in the terminal, 4 patients
pytest -q                              # 4 tests, <1 s
python -m openhospital.run_eval --name baseline
```

Dashboard (two terminals):

```bash
uvicorn api:app --port 8000
streamlit run dashboard/app.py
```

## Where each agent lives

| # | Agent | File | Type |
|---|---|---|---|
| 0 | Supervisor | `careloop/graph.py` | LangGraph StateGraph |
| 1 | Intake & Router | `careloop/agents/intake.py` | code |
| 2 | Perception workers | `careloop/agents/perception.py` | CV (mock / TorchXRayVision / TsionVision hook) |
| 3 | Report Extractor | `careloop/agents/extractor.py` | LLM (mock: regex) + verbatim-quote guard |
| 4 | Reconciler | `careloop/agents/reconciler.py` | code |
| 5 | Urgency Triage | `careloop/agents/triage.py` | rules |
| 6 | Diagnostic Reasoner | `careloop/agents/reasoner.py` | LLM (mock: rules) |
| 7 | Verifier | `careloop/agents/verifier.py` | code checks + LLM critic |
| 8 | Care Planner | `careloop/agents/planner.py` | guideline rules (`knowledge.py`) |
| 9 | Clinician Comms | `careloop/agents/clinician_comms.py` | code + LLM for EDIT |
| 10 | Patient Comms | `careloop/agents/patient_comms.py` | LLM + serious-word guard |
| 11 | Scheduler | `careloop/agents/scheduler.py` | code (mock calendar) |
| 12 | Follow-up Watchdog | `careloop/agents/watchdog.py` | code (timers) |

Shared case record: `careloop/schema.py`. Entry points: `careloop/service.py`
(`submit_scan`, `doctor_reply`, `patient_reply`, `advance_clock`).

## Turning on the real pieces (suggested order, solo)

1. **Real LLM** (~1 h): `pip install langchain langchain-anthropic`, `export ANTHROPIC_API_KEY=...`,
   set `llm.provider: anthropic` in `config.yaml`. Run `run_demo.py` and the eval. The Extractor,
   Reasoner, Verifier critic and Patient Comms switch to the LLM automatically.
2. **Real chest X-ray model** (~2 h): install torch, torchvision, torchxrayvision, scikit-image; set
   `cv.cxr_backend: torchxrayvision`; give demo scans an `image_path` to a public CXR PNG. Then add
   Grad-CAM heatmaps (TODO in `perception.py`).
3. **Photon texting** (~1-2 h): fill in `PhotonMessenger.send()` in `runtime.py`, point Photon's
   inbound webhook at `POST /reply/doctor` and `/reply/patient`, set `messenger: photon`.
4. **More OpenHospital tasks** (~2 h): mock mode scores 1.0 by construction because the mock agents
   were written for these cases. Real numbers come from the real LLM on tasks you didn't tune on:
   write ~10 more synthetic reports, mark a third `split="holdout"`.
5. **Stretch:** TsionVision brain MRI (`run_brain_mri`), MedGemma fallback, Postgres store.

## Deep Agent Diagnostic Investigator (LangChain prize)

`careloop/agents/deep_reasoner.py` replaces agent 6 with a LangChain Deep Agent when
`llm.reasoner_backend: deepagent` (needs a real provider + key). The main graph stays fixed;
only the judgment step is agentic.

- **Plans** with `write_todos` (added via `TodoListMiddleware`).
- **Tools read only the case record:** `get_timeline`, `get_finding`, `get_patient_history`,
  `compare_with_prior`, `lookup_guideline` (coded rules from `knowledge.py`).
- **Subagents:** `radiology-critic` (challenges the ranking) and `guideline-checker`.
- **Writes** `/workup.md` to its virtual filesystem; shown on the dashboard with the plan and tool steps.
- **Must finish with `submit_differential`,** which rejects invented finding IDs so the agent fixes them.
  The code Verifier still runs afterwards. If the agent never submits, it falls back to the single-call reasoner.

```bash
pip install deepagents
python -m openhospital.run_eval --name deepagent --set llm.provider=anthropic \
    --set llm.reasoner_backend=deepagent --runs 3
```

Tracing in LangSmith: `export LANGSMITH_TRACING=true LANGSMITH_API_KEY=...` (no code change).

## OpenHospital (evaluation environment)

Fresh isolated world per episode, scripted doctor/patient, code graders (0/1 checks, score = mean).
Includes anti-reward-hacking checks (a follow-up may not close without a real scan).

```bash
python -m openhospital.run_eval --name with-verifier
python -m openhospital.run_eval --name no-verifier --set pipeline.verifier_enabled=false
python -m openhospital.run_eval --name sonnet --set llm.provider=anthropic --runs 3
python -m openhospital.run_eval --split holdout --name final    # only for final numbers
```

Every run appends a row to `runs/leaderboard.csv` (score, per-task scores, seconds, LLM calls).
Rules: freeze tasks and graders before comparing; run real-LLM configs 3+ times; don't tune on holdout.

## Clinical rules caveat

`careloop/knowledge.py` holds **simplified demo rules** (Fleischner 2017 lung-nodule intervals,
critical-results list). Verify against the published guidelines before presenting them as clinical
logic. Follow-up intervals come only from code, never from an LLM.
