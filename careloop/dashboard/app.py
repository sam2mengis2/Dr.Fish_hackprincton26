"""CareLoop clinician dashboard (Streamlit). Run the API first, then:
    streamlit run dashboard/app.py
Set CARELOOP_API if the API isn't on localhost:8000.
"""
from __future__ import annotations

import html
import json as _json
import os
from pathlib import Path

import httpx
import streamlit as st

API = os.environ.get("CARELOOP_API", "http://localhost:8000")
st.set_page_config(page_title="CareLoop", page_icon="🩺", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
try:
    DEMO_SCANS = _json.loads((ROOT / "data" / "demo_cases.json").read_text())["scans"]
except Exception:
    DEMO_SCANS = []


def get(path):
    return httpx.get(API + path, timeout=60).json()


def post(path, body=None):
    r = httpx.post(API + path, json=body or {}, timeout=600)
    if r.status_code >= 400:
        st.error(r.text)
    return r


# ------------------------------------------------------------------ look and feel
st.markdown("""
<style>
:root { --teal:#14b8a6; --teal2:#0d9488; --card:#111a2e; --line:#1f2a44; --muted:#8b98b3; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.01em; }
.cl-hero { display:flex; align-items:center; justify-content:space-between; padding:14px 20px; margin-bottom:14px;
  border:1px solid var(--line); border-radius:14px;
  background: linear-gradient(120deg, rgba(20,184,166,.18), rgba(59,130,246,.10) 60%, rgba(17,26,46,.6)); }
.cl-hero .logo { font-size:1.6rem; font-weight:800; }
.cl-hero .logo span { color:var(--teal); }
.cl-hero .tag { color:var(--muted); font-size:.92rem; margin-top:2px; }
.cl-chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
  border:1px solid var(--line); background:#0f172a; margin-right:6px; white-space:nowrap; }
.chip-critical, .chip-stat { background:rgba(239,68,68,.15); color:#fca5a5; border-color:rgba(239,68,68,.45); }
.chip-urgent { background:rgba(249,115,22,.15); color:#fdba74; border-color:rgba(249,115,22,.45); }
.chip-routine { background:rgba(34,197,94,.13); color:#86efac; border-color:rgba(34,197,94,.4); }
.chip-incidental { background:rgba(59,130,246,.15); color:#93c5fd; border-color:rgba(59,130,246,.45); }
.chip-state { background:rgba(20,184,166,.12); color:#5eead4; border-color:rgba(20,184,166,.4); }
.chip-high { background:rgba(239,68,68,.12); color:#fca5a5; border-color:rgba(239,68,68,.35); }
.chip-moderate { background:rgba(234,179,8,.12); color:#fde047; border-color:rgba(234,179,8,.35); }
.chip-low { background:rgba(148,163,184,.12); color:#cbd5e1; border-color:rgba(148,163,184,.35); }
.cl-card { border:1px solid var(--line); border-radius:14px; padding:14px 18px; background:var(--card); margin-bottom:12px; }
.cl-patient .name { font-size:1.45rem; font-weight:750; }
.cl-patient .meta { color:var(--muted); font-size:.9rem; margin-top:4px; }
.cl-steps { display:flex; gap:6px; margin:6px 0 14px 0; flex-wrap:wrap; }
.cl-step { flex:1; min-width:92px; text-align:center; padding:7px 4px; border-radius:10px; font-size:.75rem; font-weight:600;
  border:1px solid var(--line); color:var(--muted); background:#0f172a; }
.cl-step.done { color:#5eead4; border-color:rgba(20,184,166,.45); background:rgba(20,184,166,.10); }
.cl-step.now { color:#0b1220; background:var(--teal); border-color:var(--teal); }
.cl-phone { border:1px solid var(--line); border-radius:22px; padding:12px 12px 6px 12px; background:#0a0f1c; margin-bottom:10px; }
.cl-phone .title { text-align:center; color:var(--muted); font-size:.8rem; font-weight:600; margin-bottom:8px; }
.cl-bubble { max-width:92%; padding:8px 12px; border-radius:16px; margin:6px 0; font-size:.86rem; line-height:1.35; }
.cl-in { background:#1e293b; color:#e2e8f0; border-bottom-left-radius:4px; }
.cl-out { background:#2563eb; color:white; margin-left:auto; border-bottom-right-radius:4px; }
.cl-alert { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.35); color:#fecaca; }
.cl-time { font-size:.68rem; color:var(--muted); margin:0 4px; }
.cl-section { font-size:1.08rem; font-weight:700; margin:14px 0 6px 0; }
.cl-note { color:var(--muted); font-size:.82rem; }
div[data-testid="stMetric"] { border:1px solid var(--line); border-radius:14px; padding:10px 14px; background:var(--card); }
section[data-testid="stSidebar"] .stButton button { width:100%; }
</style>
""", unsafe_allow_html=True)

URG = {"critical": "Critical", "urgent": "Urgent", "routine": "Routine", "incidental": "Incidental"}
STATE = {"ingested": "Received", "perceived": "Image read", "reconciled": "Findings merged", "triaged": "Triaged",
         "reasoned": "Reasoned", "verified": "Verified", "verify_failed": "Re-checking",
         "needs_human_review": "Needs human review", "planned": "Planned", "awaiting_approval": "Awaiting doctor",
         "approved": "Approved", "edited": "Edited", "rejected": "Rejected", "escalated": "Escalated",
         "doctor_calling": "Doctor calling", "active": "Loop active", "awaiting_report": "Waiting for report",
         "closed": "Loop closed"}


def chip(text, kind=""):
    return f'<span class="cl-chip chip-{kind}">{html.escape(str(text))}</span>'


def bubbles(msgs, title, reply_role):
    out = [f'<div class="cl-phone"><div class="title">{title}</div>']
    if not msgs:
        out.append('<div class="cl-note" style="text-align:center;padding:8px">No messages yet</div>')
    for m in msgs:
        cls = "cl-out" if m.get("direction") == "in" else ("cl-alert" if m["to"] == "radiologist" else "cl-in")
        when = (m.get("at") or "").replace("T", " ")
        body = html.escape(m["text"]).replace("\n", "<br>")
        out.append(f'<div class="cl-bubble {cls}">{body}</div><div class="cl-time" '
                   f'style="text-align:{"right" if cls == "cl-out" else "left"}">{when}</div>')
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar: worklist + demo controls
clock = get("/clock")["now"]
cases = get("/cases")
with st.sidebar:
    st.markdown('<div style="font-size:1.5rem;font-weight:800">Care<span style="color:#14b8a6">Loop</span></div>'
                f'<div class="cl-note">Simulated time · {clock.replace("T", " ")}</div>', unsafe_allow_html=True)
    st.markdown("#### Worklist")
    pid = None
    if cases:
        dot = {"critical": "🔴", "urgent": "🟠", "routine": "🟢", "incidental": "🔵"}
        labels = [f"{dot.get(c['urgency'], '⚪')} {c['name']}  ·  {STATE.get(c['state'], c['state'])}" for c in cases]
        idx = st.radio("Most urgent first", range(len(cases)), format_func=lambda i: labels[i],
                       label_visibility="collapsed")
        pid = cases[idx]["patient_id"]
    else:
        st.caption("No patients yet.")
    st.divider()
    st.markdown("#### Demo controls")
    if pid and not pid.startswith("P-1"):
        if st.button(f"▶  Next step for {pid}", type="primary", help="Sign the pending report, or bring the follow-up scan"):
            post(f"/demo/load/{pid}")
            st.rerun()
    loaded = {c["patient_id"] for c in cases}
    if st.button("🌅  Simulate morning", help="Busy scanners + 3 scans waiting for radiologist reports"):
        post("/demo/busy")
        for d in ("P-004", "P-002", "P-003"):  # arrival order; the queue re-sorts by need
            if d not in loaded:                # already in today's demo: skip instead of erroring
                post(f"/demo/load/{d}")
                post("/clock/advance", {"minutes": 5})
        st.rerun()
    if st.button("📭  Simulate a cancellation", help="P-101 gives up their Dec 3 slot"):
        post("/demo/cancel"); st.rerun()
    others = [p for p in ("P-001", "P-002", "P-003", "P-004") if p not in loaded]
    if others:
        new_pid = st.selectbox("Load a new patient", others)
        if st.button("Load first scan"):
            post(f"/demo/load/{new_pid}")
            st.rerun()
    st.caption("Advance the clock")
    c1, c2, c3 = st.columns(3)
    if c1.button("+15m"):
        post("/clock/advance", {"minutes": 15}); st.rerun()
    if c2.button("+1d"):
        post("/clock/advance", {"days": 1}); st.rerun()
    if c3.button("+30d"):
        post("/clock/advance", {"days": 30}); st.rerun()
    st.divider()
    if st.button("↺  Reset demo"):
        post("/reset"); st.rerun()

# ------------------------------------------------------------------ hero + impact numbers
st.markdown('<div class="cl-hero"><div><div class="logo">Care<span>Loop</span></div>'
            '<div class="tag">From scan to completed follow-up: nothing waits, nothing gets lost.</div></div>'
            f'<div>{chip("Doctor approves every plan", "state")}{chip("Guidelines in code, not AI", "routine")}</div></div>',
            unsafe_allow_html=True)

m = get("/metrics")
if m and (m.get("reports_signed") or m.get("days_saved") or m.get("unsigned")):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reports signed / waiting", f"{m['reports_signed']} / {m['unsigned']}",
              help="Unsigned scans are ranked in the reading queue and chased when overdue")
    c2.metric("Days saved by cancellations", m["days_saved"],
              f"{m['patients_moved_up']} moved up · {m['back_on_time']} back on time")
    c3.metric("Human actions", m["human_actions"], f"{m['automated_steps']} automated steps", delta_color="off",
              help="Doctor/patient replies vs texts, bookings and reminders CareLoop did itself")
    c4.metric("Deep Agent model calls", m["deep_agent_calls"] or "-",
              help="All model calls inside the LangChain Deep Agent, subagents included")

queue = get("/reading-queue")
wl = get("/waitlist")
n_late = sum(r["late"] for r in wl)
tab_pt, tab_q, tab_b, tab_log = st.tabs([
    "🩺  Patient", f"🩻  Reading queue ({len(queue)})", f"📅  Scanner bookings ({n_late} late)", "📜  Audit log"])

# ------------------------------------------------------------------ reading queue
with tab_q:
    st.markdown('<div class="cl-note">Sorted by need, not arrival: order reason, open follow-ups and the chest X-ray '
                'model. The AI pre-read only sets the order; it is never sent to the family doctor.</div>',
                unsafe_allow_html=True)
    if not queue:
        st.success("No unsigned reports. The queue is clear.")
    PRI = {"stat": "🔴 STAT", "urgent": "🟠 Urgent", "routine": "🟢 Routine"}
    if queue:
        st.dataframe([{"Priority": PRI.get(r["priority"], r["priority"]), "Patient": f"{r['patient_id']} {r['name']}",
                       "Exam": r["exam"], "Site": r["site"], "Order reason": r["indication"],
                       "Why this position": "; ".join(r["reasons"]), "Waiting (h)": r["waiting_hours"],
                       "Overdue": "⚠️" if r["overdue"] else ""} for r in queue], hide_index=True, width='stretch')

# ------------------------------------------------------------------ bookings / cancellation list
with tab_b:
    st.markdown('<div class="cl-note">Ordered by clinical due date. A freed slot is offered by text to the patient who '
                'needs it soonest; each accepted offer frees a later slot for the next patient.</div>',
                unsafe_allow_html=True)
    if not wl:
        st.info("No bookings yet. Approve a plan, or click Simulate morning.")
    else:
        st.dataframe([{"Patient": f"{r['patient_id']} {r['name']}", "Scan": r["action"], "Due (guideline)": r["due"],
                       "Booked": r["appointment"].replace("T", " "), "Site": r["site"],
                       "Status": ("📨 offered " + r["offer"]["slot"].replace("T", " ")) if r["offer"]
                       else ("⚠️ later than due · cancellation list" if r["late"] else "✅ on time"),
                       "Why": r["note"]} for r in wl], hide_index=True, width='stretch')

if not pid:
    with tab_pt:
        st.info("No patients yet. Click **Simulate morning** or **Load a new patient** in the sidebar.")
    st.stop()

case = get(f"/cases/{pid}")
p = case["patient"]

with tab_log:
    for e in reversed(case["audit_log"]):
        st.text(f"{e['at']}  [{e['agent']}]  {e['action']}")

# ------------------------------------------------------------------ patient view
with tab_pt:
    urg = case.get("urgency")
    st.markdown(
        f'<div class="cl-card cl-patient"><div class="name">{html.escape(p["name"])} '
        f'<span class="cl-note" style="font-size:1rem">{p["patient_id"]} · {p["age"]}{p["sex"]}</span></div>'
        f'<div style="margin-top:6px">{chip(URG.get(urg, "Not triaged"), urg or "")}'
        f'{chip(STATE.get(case["state"], case["state"]), "state")}</div>'
        f'<div class="meta">Ordering: {html.escape(p["ordering_doctor"])} · History: '
        f'{html.escape("; ".join(p["history"]) or "none recorded")}'
        + (f' · Prefers: {html.escape(p["preferred_site"])}' if p.get("preferred_site") else "") + '</div></div>',
        unsafe_allow_html=True)

    # where this case is in the loop
    order = ["Scan", "Report", "AI review", "Doctor", "Booked", "Follow-up"]
    s = case["state"]
    stage = {"awaiting_report": 1, "ingested": 1, "perceived": 1, "reconciled": 2, "triaged": 2, "reasoned": 2,
             "verified": 2, "verify_failed": 2, "needs_human_review": 3, "planned": 3, "awaiting_approval": 3,
             "escalated": 3, "doctor_calling": 3, "approved": 4, "edited": 4, "rejected": 3, "active": 4,
             "closed": 6}.get(s, 0)
    if s == "active" and any(f["status"] == "completed" and f.get("completed_by_scan") for f in case["followups"]):
        stage = 5
    st.markdown('<div class="cl-steps">' + "".join(
        f'<div class="cl-step {"done" if i < stage else "now" if i == stage else ""}">{name}</div>'
        for i, name in enumerate(order)) + "</div>", unsafe_allow_html=True)

    pend = [x for x in case["scans"] if x.get("report_status") == "pending"]
    if pend:
        s0 = pend[-1]
        PRI = {"stat": "🔴 STAT", "urgent": "🟠 Urgent", "routine": "🟢 Routine"}
        st.info(f"⏳ **Waiting for the radiologist's report** on {s0['modality']} {s0['body_part'].lower()} "
                f"({s0.get('site') or 'this site'}), taken {(s0.get('acquired_at') or '').replace('T', ' ')}.  \n"
                f"Reading queue: **{PRI.get(s0.get('read_priority'), s0.get('read_priority'))}** because "
                f"{'; '.join(s0.get('read_priority_reasons') or [])}.  \n"
                "Nothing is shown to the doctor or patient until the report is signed; the AI pre-read only sets "
                "reading order. Click **Next step** in the sidebar to sign the demo report.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="cl-section">Scans</div>', unsafe_allow_html=True)
        st.dataframe([{"Scan": x["scan_id"], "Date": x["date"], "Exam": f"{x['modality']} {x['body_part'].lower()}",
                       "Site": x.get("site", "") + (" (outside record)" if x.get("outside") else ""),
                       "Order reason": x.get("indication", ""),
                       "Report": "⏳ pending" if x.get("report_status") == "pending" else "✅ signed"}
                      for x in case["scans"]], hide_index=True, width='stretch')
        signed = [x for x in case["scans"] if x.get("report_status") != "pending" and x.get("report_text")]
        if signed:
            with st.expander(f"📄  View reports ({len(signed)} signed)"):
                for x in signed:
                    who = " (outside record)" if x.get("outside") else ""
                    st.markdown(f"**{x['scan_id']}** · {x['modality']} {x['body_part'].lower()} · {x['date']} · "
                                f"{x.get('site') or ''}{who}")
                    st.markdown(f"> {html.escape(x['report_text'])}")
        # The images themselves, from the moment they exist: the radiologist reads them before signing,
        # and the doctor can open them later. Scans with no image file are report-only in this demo.
        def _img(x):
            ip = x.get("image_path")
            if not ip:
                return None
            fp = Path(ip)
            fp = fp if fp.is_absolute() else ROOT / fp
            return fp if fp.exists() else None
        with_img = [(x, _img(x)) for x in case["scans"]]
        shown = [(x, fp) for x, fp in with_img if fp]
        missing = [x for x, fp in with_img if x.get("image_path") and not fp]
        if shown:
            st.markdown('<div class="cl-note" style="margin-top:8px"><b>Images</b></div>', unsafe_allow_html=True)
            cols = st.columns(min(len(shown), 3))
            for col, (x, fp) in zip(cols, shown):
                state = "report pending: radiologist to read" if x.get("report_status") == "pending" else "report signed"
                col.image(str(fp), width=300, caption=f"{x['scan_id']} · {x['modality']} {x['body_part'].lower()} · "
                                                     f"{x['date']} · {state}"
                                                     + (f"  \nImage: {x['image_credit']}" if x.get("image_credit") else ""))
        for x in missing:
            st.caption(f"Image for {x['scan_id']} not found ({x['image_path']}). Add the file to show it here.")
        if pend:
            with st.expander("✍️  Sign a report as the radiologist"):
                with st.form("sign"):
                    ps = st.selectbox("Scan", [x["scan_id"] for x in pend])
                    demo_rep = next((e["scan"]["report_text"] for e in DEMO_SCANS if e["scan"]["scan_id"] == ps), "")
                    rep = st.text_area("Report text (prefilled with the demo report; edit it to try a variant)",
                                       demo_rep)
                    if st.form_submit_button("Sign report") and rep.strip():
                        post("/reports/sign", {"patient_id": pid, "scan_id": ps, "report_text": rep}); st.rerun()

        # Before the report is signed there is nothing new to show; earlier/outside findings are still history.
        show_results = not pend or bool(case["differential"]) or bool(case["followups"])
        if not show_results and case["findings"]:
            st.markdown('<div class="cl-section">Earlier findings (signed reports, incl. other hospitals)</div>',
                        unsafe_allow_html=True)
            st.dataframe([{"ID": f["finding_id"], "Finding": f["finding"], "Location": f["location"],
                           "Size (mm)": f["size_mm"], "Scan": f["scan_id"], "Date": f["date"]}
                          for f in case["findings"]], hide_index=True, width='stretch')
        if show_results:
            st.markdown('<div class="cl-section">Findings</div>', unsafe_allow_html=True)
            st.dataframe([{"ID": f["finding_id"], "Finding": f["finding"], "Location": f["location"],
                           "Size (mm)": f["size_mm"], "Conf": f["confidence"],
                           "Sources": " + ".join(f["sources"]), "Scan": f["scan_id"], "Date": f["date"],
                           "Same lesion as": f.get("linked_to") or "", "Growth (mm)": f.get("growth_mm")}
                          for f in case["findings"]], hide_index=True, width='stretch')
            heats = [f for f in case["findings"] if f.get("heatmap_path") and Path(f["heatmap_path"]).exists()]
            if heats:
                st.markdown('<div class="cl-note" style="margin-top:8px"><b>Where the X-ray model looked</b> '
                            '(Grad-CAM, open-source TorchXRayVision)</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(heats), 3))
                for col, f in zip(cols, heats):
                    col.image(f["heatmap_path"], width=300,
                              caption=f"{f['finding_id']} {f['finding']} (model {f['confidence']:.2f}). "
                                      "Shows the model's attention, not a diagnosis.")
            if case["discrepancies"]:
                st.warning("**Second look needed:** " + " | ".join(
                    f"{d['finding_id']}: {d['note']}" for d in case["discrepancies"]))

            st.markdown('<div class="cl-section">Possible diagnoses <span class="cl-note">for physician review, '
                        'not a diagnosis</span></div>', unsafe_allow_html=True)
            v = case["verifier"]
            if case.get("urgency") == "critical" and not case["differential"]:
                st.warning("**Critical fast path:** diagnosis reasoning skipped on purpose. The doctor was alerted "
                           "immediately; speed matters more than a differential here.")
            else:
                (st.success if v["status"] == "pass" else st.error if v["status"] == "fail" else st.info)(
                    f"Verifier: {v['status']} after {v['attempts']} attempt(s). {' '.join(v['notes'])}")
            if case["insufficient_info"]:
                st.info("Insufficient information to suggest a diagnosis.")
            for d in case["differential"]:
                with st.expander(f"#{d['rank']}  {d['dx']}   ·   {d['likelihood']}", expanded=d["rank"] == 1):
                    st.markdown(chip(f"{d['likelihood']} likelihood", d["likelihood"])
                                + chip("Evidence: " + (", ".join(d["evidence_for"]) or "none cited"))
                                + chip("Contradicting: " + (", ".join(d["evidence_against"]) or "none in the scans")),
                                unsafe_allow_html=True)
                    st.write(d["rationale"])
                    if d.get("caveats"):
                        st.markdown("**Against / uncertainties**\n" + "\n".join(f"- {c}" for c in d["caveats"]))
                    if d["next_tests"]:
                        st.markdown("**Next tests:** " + ", ".join(d["next_tests"]))
            if v.get("suggestions"):
                with st.expander(f"Reviewer notes ({len(v['suggestions'])}, non-blocking)"):
                    for sgg in v["suggestions"]:
                        st.markdown(f"- {sgg}")
            if case.get("agent_trace") or case.get("workup_note"):
                with st.expander("🧠  Deep Agent investigation (LangChain Deep Agents)"):
                    a, b = st.columns(2)
                    if case.get("agent_plan"):
                        a.markdown("**Plan (write_todos)**\n\n" + "\n".join(f"- {t}" for t in case["agent_plan"]))
                    if case.get("agent_trace"):
                        b.markdown("**Tool steps**\n\n" + "\n".join(
                            f"{i + 1}. `{t}`" for i, t in enumerate(case["agent_trace"])))
                    if case.get("workup_note"):
                        st.markdown("**Workup note (/workup.md)**")
                        st.markdown(case["workup_note"])

            st.markdown('<div class="cl-section">Follow-up plan</div>', unsafe_allow_html=True)
            STAT = {"booked": "📅 booked", "completed": "✅ done", "missed": "⚠️ missed", "escalated": "🚨 escalated",
                    "open": "🕓 open", "patient_notified": "📨 notified", "cancelled": "✖ cancelled"}
            st.dataframe([{"Action": f["action"], "Due": f["due"], "Status": STAT.get(f["status"], f["status"]),
                           "Appointment": (f["appointment"] or "").replace("T", " "), "Site": f.get("site") or "",
                           "Owner": f["owner"], "Rule (why this date)": f["rule"]}
                          for f in case["followups"]], hide_index=True, width='stretch')

        open_items = [f for f in case["followups"] if f["status"] not in ("completed", "cancelled")]
        if open_items:
            with st.expander("Record a result (e.g. lab or visit done)"):
                with st.form("result"):
                    pick = st.selectbox("Result received for", [f"{f['id']} - {f['action']}" for f in open_items])
                    note = st.text_input("Result note", "normal")
                    if st.form_submit_button("Record result"):
                        post("/results", {"patient_id": pid, "followup_id": pick.split(" - ")[0], "note": note})
                        st.rerun()

    with right:
        rad = [x for x in case["messages"] if x["to"] == "radiologist"]
        if rad:
            bubbles(rad[-3:], "📟  Radiology alerts", None)
        bubbles([x for x in case["messages"] if x["to"] in ("doctor", "covering_doctor")][-5:],
                f"📱  {html.escape(p['ordering_doctor'])}'s phone", "doctor")
        with st.form("doc"):
            c1, c2 = st.columns([3, 1])
            t = c1.text_input("Reply as doctor", "APPROVE", label_visibility="collapsed")
            if c2.form_submit_button("Send"):
                post("/reply/doctor", {"patient_id": pid, "text": t}); st.rerun()
        bubbles([x for x in case["messages"] if x["to"] == "patient"][-5:],
                f"📱  {html.escape(p['name'].split()[0])}'s phone", "patient")
        with st.form("pt"):
            c1, c2 = st.columns([3, 1])
            t = c1.text_input("Reply as patient", "OK", label_visibility="collapsed",
                              help="OK, RESCHEDULE, SWITCH, TAKE, SKIP, CANCEL, or a question")
            if c2.form_submit_button("Send"):
                post("/reply/patient", {"patient_id": pid, "text": t}); st.rerun()
        st.caption("Patient replies: OK · RESCHEDULE · SWITCH · TAKE · SKIP · CANCEL · or ask a question")
