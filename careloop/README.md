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
| 2 | Perception workers | `careloop/agents/perception.py` | CV (mock / TorchXRayVision on real chest X-rays; MRI/CT hooks) |
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
5. **Stretch:** open-source brain MRI / CT models in `run_brain_mri` and the fallback slot, Postgres store.

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

## Photon 2.0 texting (doctor approvals + patient updates over iMessage)

`careloop/photon_messenger.py` uses Photon's Python client (`photonhq-api`). Replies are **polled**
every 2 s, so no public URL or tunnel is needed.

1. Photon Beta dashboard (app.beta.photon.codes): create/select a project; **API keys -> New API key** with
   `project:read` + `project:write`.
2. **iMessage platform page:** enter your iPhone's number as the sender, confirm it on the phone
   (Settings -> Apps -> Messages -> Send & Receive), then **Add** to get an assigned shared-line number.
3. Environment (setx, never in git): `PHOTON_API_KEY`, `PHOTON_PROJECT_ID`, `PHOTON_AGENT_HANDLE` (the assigned number).
4. `config.yaml`: `messenger: photon`, then `uvicorn api:app --port 8000` and check `GET /photon/status`.
5. From the phone, text the assigned number **`DOCTOR PATIENT`** (solo: one phone plays both roles) or
   `DOCTOR` / `PATIENT` from two phones. Photon's shared line can't text first, so this step is required.
6. Load a demo scan from the dashboard: the doctor alert arrives as an iMessage. Reply **APPROVE**; the
   patient's booking text arrives. Prefix a patient ID to target a case: `P-002 APPROVE`.

## Imaging-department pain points (from Leyat Mesfin, imaging, Mount Sinai)

1. **Reports are delayed**, so the family doctor can't see results.
   - `service.acquire_scan`: images exist, report not signed. The scan joins the **radiologist reading queue**
     (`GET /reading-queue`), ranked stat / urgent / routine by `agents/read_priority.py` from the order reason,
     open follow-ups (e.g. a growing nodule on a 3-month deadline), and the chest X-ray model's pre-read.
     An LLM read of the order reason may move a scan UP, never down. The pre-read only sets reading order;
     nothing is sent to the family doctor or patient before the report is signed.
   - STAT scans alert the radiologist immediately. Watchdog `report_overdue` alerts radiology at the target
     (`timers.report_due_hours`), then the radiology lead + the ordering doctor ("delayed, you'll hear when signed").
   - `service.sign_report`: the late report attaches to the same scan and the normal loop runs; the doctor's
     text opens with "Report ready".
2. **Cross-hospital data and scheduling.**
   - `service.import_outside_record`: a prior scan from another hospital becomes history, so a new nodule is
     compared against it ("same lesion as the 2024 CT at Riverside, then 4 mm (+3 mm)"). Growth >= 2 mm
     triggers the growing-nodule rule (CT within 3 months instead of 6-12).
   - One calendar per hospital: follow-ups book at the patient's `preferred_site`; the patient can reply
     SWITCH to move to another hospital; a follow-up scan done at either hospital closes the loop.

3. **Patients wait 5-6 months for CT/MRI.** CareLoop can't add scanners; it stops slots being wasted and
   sends patients to the shortest line (`agents/scheduler.py`, config `scheduling`):
   - Books the patient's preferred hospital only if it meets the guideline due date, otherwise the earliest
     slot across the network. Anyone booked later than their due date is on the **cancellation list**.
   - A freed slot (cancellation, reschedule, an accepted offer) is offered by text to the waiting patient with
     the **most urgent clinical due date** ("reply TAKE / SKIP", held `offer_hours`), then the next. Each accepted
     offer frees a later slot, which is offered on, so one cancellation can put several patients back on time.
   - `GET /waitlist`, `POST /cancel`. Patients reply TAKE, SKIP, CANCEL, RESCHEDULE or SWITCH.
   - OpenHospital T10-T12 grade the reading queue, overdue-report chasing and cancellation backfill
     (each with an anti-reward-hacking check; deliberately breaking the ordering makes them fail).

Real image model: set `cv.cxr_backend: torchxrayvision` (`pip install torchxrayvision`; weights download on first
use). Demo X-rays in `data/scans/` are public NIH ChestX-ray14 sample images from the TorchXRayVision repo.
In real-model mode a scan without an image gets no image read (no made-up outputs). Each flagged finding
gets a **Grad-CAM heatmap** (last dense block of the DenseNet) saved to `runs/heatmaps/` and shown on the
dashboard: where the model's evidence came from, not a diagnosis. James's demo report says *left* hilum
because that is where the real image's opacity (and the model's attention) is. CT/MRI are read from the
report only; open-source CT/MRI models would plug into `perception.py`.

Dashboard demo: sidebar "Simulate morning" (busy scanners + 3 scans arrive; the pneumothorax jumps the
queue), then "Next step" for P-002 signs Maria's report -> doctor text -> APPROVE -> Maria is booked at
Riverside in March, later than her February due date, so she's on the cancellation list -> "Simulate a
cancellation" -> Maria is texted the December slot -> reply TAKE -> her March slot passes to Tom, who is now
on time too -> "Next step" again = her follow-up CT arrives and closes the loop.

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
