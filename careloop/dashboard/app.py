"""CareLoop clinician dashboard (Streamlit). Run the API first, then:
    streamlit run dashboard/app.py
Set CARELOOP_API if the API isn't on localhost:8000.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

API = os.environ.get("CARELOOP_API", "http://localhost:8000")
st.set_page_config(page_title="CareLoop", layout="wide")


def get(path):
    return httpx.get(API + path, timeout=600).json()


try:
    import json as _json
    from pathlib import Path as _Path

    DEMO_SCANS = _json.loads((_Path(__file__).resolve().parents[1] / "data" / "demo_cases.json").read_text())["scans"]
except Exception:
    DEMO_SCANS = []


def post(path, body=None):
    r = httpx.post(API + path, json=body or {}, timeout=600)
    if r.status_code >= 400:
        st.error(r.text)
    return r


URG_COLOR = {"critical": "🔴", "urgent": "🟠", "routine": "🟢", "incidental": "🔵", None: "⚪"}

# ------------------------------------------------------------------ sidebar: worklist + demo controls
with st.sidebar:
    st.title("CareLoop")
    st.caption(f"Simulated time: {get('/clock')['now']}")
    st.subheader("Worklist")
    cases = get("/cases")
    pid = None
    if cases:
        labels = [f"{URG_COLOR[c['urgency']]} {c['patient_id']} {c['name']} - {c['state']}" for c in cases]
        idx = st.radio("Cases (most urgent first)", range(len(cases)), format_func=lambda i: labels[i])
        pid = cases[idx]["patient_id"]
    st.divider()
    st.subheader("Demo controls")
    if pid and not pid.startswith("P-1"):
        if st.button(f"Next step for {pid} (sign report / follow-up scan)", type="primary"):
            post(f"/demo/load/{pid}")
            st.rerun()
    loaded = {c["patient_id"] for c in cases}
    if st.button("Simulate morning: busy scanners + 3 scans waiting for reports"):
        post("/demo/busy")
        for d in ("P-004", "P-002", "P-003"):  # arrival order; the queue re-sorts by need
            if d not in loaded:                # already in today's demo: skip instead of erroring
                post(f"/demo/load/{d}")
                post("/clock/advance", {"minutes": 5})
        st.rerun()
    others = [p for p in ("P-001", "P-002", "P-003", "P-004") if p not in loaded]
    if others:
        new_pid = st.selectbox("Load a new patient", others)
        if st.button("Load first scan"):
            post(f"/demo/load/{new_pid}")
            st.rerun()
    c1, c2, c3 = st.columns(3)
    if c1.button("+15 min"):
        post("/clock/advance", {"minutes": 15}); st.rerun()
    if c2.button("+1 day"):
        post("/clock/advance", {"days": 1}); st.rerun()
    if c3.button("+30 days"):
        post("/clock/advance", {"days": 30}); st.rerun()
    if st.button("Simulate a cancellation (P-101 gives up Dec 3 slot)"):
        post("/demo/cancel"); st.rerun()
    if st.button("Reset demo"):
        post("/reset"); st.rerun()

queue = get("/reading-queue")
if queue:
    st.subheader(f"Radiologist reading queue ({len(queue)} unsigned)")
    st.caption("Sorted by need, not arrival: order reason + open follow-ups + image model (chest X-ray). "
               "The AI pre-read only sets the order; it is never sent to the family doctor.")
    PRI = {"stat": "🔴 STAT", "urgent": "🟠 Urgent", "routine": "🟢 Routine"}
    st.dataframe([{"Priority": PRI.get(r["priority"], r["priority"]), "Patient": f"{r['patient_id']} {r['name']}",
                   "Exam": r["exam"], "Site": r["site"], "Order reason": r["indication"],
                   "Why this position": "; ".join(r["reasons"]), "Waiting (h)": r["waiting_hours"],
                   "Overdue": "⚠️" if r["overdue"] else ""} for r in queue], hide_index=True, width='stretch')
    st.divider()

wl = get("/waitlist")
if wl:
    with st.expander(f"Scanner bookings & cancellation list ({sum(r['late'] for r in wl)} booked later than due)",
                     expanded=any(r["late"] or r["offer"] for r in wl)):
        st.caption("Ordered by clinical due date. A freed slot is offered by text to the first patient who "
                   "needs it sooner; each accepted offer frees a later slot for the next.")
        st.dataframe([{"Patient": f"{r['patient_id']} {r['name']}", "Scan": r["action"], "Due (guideline)": r["due"],
                       "Booked": r["appointment"].replace("T", " "), "Site": r["site"],
                       "Status": ("📨 offered " + r["offer"]["slot"].replace("T", " ")) if r["offer"]
                       else ("⚠️ later than due - on cancellation list" if r["late"] else "✅ on time"),
                       "Why": r["note"]} for r in wl], hide_index=True, width='stretch')

if not pid:
    st.info("No cases yet. Load a demo scan from the sidebar.")
    st.stop()

case = get(f"/cases/{pid}")
p = case["patient"]
st.header(f"{URG_COLOR[case['urgency']]} {p['name']} ({p['patient_id']}), {p['age']}{p['sex']}")
st.caption(f"State: **{case['state']}** | Ordering: {p['ordering_doctor']} | History: {'; '.join(p['history'])}")

pend = [s for s in case["scans"] if s.get("report_status") == "pending"]
if pend:
    s0 = pend[-1]
    PRI = {"stat": "🔴 STAT", "urgent": "🟠 Urgent", "routine": "🟢 Routine"}
    st.info(f"⏳ **Waiting for the radiologist's report** on {s0['modality']} {s0['body_part'].lower()} "
            f"({s0.get('site') or 'this site'}), taken {(s0.get('acquired_at') or '').replace('T', ' ')}.  \n"
            f"Reading queue: **{PRI.get(s0.get('read_priority'), s0.get('read_priority'))}** because "
            f"{'; '.join(s0.get('read_priority_reasons') or [])}.  \n"
            "No findings, diagnoses or plan are shown to the doctor or patient until the report is signed; "
            "the AI pre-read only sets reading order. Click **Next step** in the sidebar to sign the demo report.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Scans")
    st.dataframe([{"Scan": s["scan_id"], "Date": s["date"], "Exam": f"{s['modality']} {s['body_part'].lower()}",
                   "Site": s.get("site", "") + (" (outside record)" if s.get("outside") else ""),
                   "Order reason": s.get("indication", ""),
                   "Report": "⏳ pending" if s.get("report_status") == "pending" else "✅ signed"}
                  for s in case["scans"]], hide_index=True, width='stretch')
    pending = [s for s in case["scans"] if s.get("report_status") == "pending"]
    if pending:
        with st.form("sign"):
            ps = st.selectbox("Sign report for", [s["scan_id"] for s in pending])
            demo_rep = next((e["scan"]["report_text"] for e in DEMO_SCANS if e["scan"]["scan_id"] == ps), "")
            rep = st.text_area("Report text (radiologist) - prefilled with the demo report; edit to try a variant",
                               demo_rep)
            if st.form_submit_button("Sign report") and rep.strip():
                post("/reports/sign", {"patient_id": pid, "scan_id": ps, "report_text": rep}); st.rerun()

    # Before the report is signed there is nothing new to show; earlier/outside findings are still history.
    show_results = not pend or bool(case["differential"]) or bool(case["followups"])
    if not show_results and case["findings"]:
        st.subheader("Earlier findings (signed reports, incl. other hospitals)")
        st.dataframe([{"ID": f["finding_id"], "Finding": f["finding"], "Location": f["location"],
                       "Size (mm)": f["size_mm"], "Scan": f["scan_id"], "Date": f["date"]}
                      for f in case["findings"]], hide_index=True, width='stretch')
    if show_results:
        st.subheader("Findings")
        rows = [{"ID": f["finding_id"], "Finding": f["finding"], "Location": f["location"],
                 "Size (mm)": f["size_mm"], "Count": f["count"], "Conf": f["confidence"],
                 "Sources": " + ".join(f["sources"]), "Scan": f["scan_id"], "Date": f["date"],
                 "Same lesion as": f.get("linked_to") or "", "Growth (mm)": f.get("growth_mm")}
                for f in case["findings"]]
        st.dataframe(rows, hide_index=True, width='stretch')
        for f in case["findings"]:
            if f.get("heatmap_path") and Path(f["heatmap_path"]).exists():
                st.image(f["heatmap_path"], caption=f"{f['finding_id']} heatmap", width=260)
        if case["discrepancies"]:
            st.warning("**Second look needed:** " + " | ".join(f"{d['finding_id']}: {d['note']}" for d in case["discrepancies"]))

        st.subheader("Possible diagnoses (for physician review)")
        if case["insufficient_info"]:
            st.info("Insufficient information to suggest a diagnosis.")
        for d in case["differential"]:
            with st.expander(f"#{d['rank']} {d['dx']}  [{d['likelihood']}]", expanded=d["rank"] == 1):
                st.write(f"**Evidence for:** {', '.join(d['evidence_for']) or '-'}   "
                         f"**Against:** {', '.join(d['evidence_against']) or '-'}")
                st.write(d["rationale"])
                st.write("**Next tests:** " + ", ".join(d["next_tests"]))
        if case.get("agent_trace") or case.get("workup_note"):
            with st.expander("Deep Agent investigation (LangChain Deep Agents)", expanded=False):
                if case.get("agent_plan"):
                    st.markdown("**Plan (write_todos)**\n\n" + "\n".join(f"- {t}" for t in case["agent_plan"]))
                if case.get("agent_trace"):
                    st.markdown("**Tool steps**\n\n" + "\n".join(f"{i + 1}. `{t}`" for i, t in enumerate(case["agent_trace"])))
                if case.get("workup_note"):
                    st.markdown("**Workup note (/workup.md)**")
                    st.markdown(case["workup_note"])
        v = case["verifier"]
        if case.get("urgency") == "critical" and not case["differential"]:
            st.warning("Critical fast path: diagnosis reasoning skipped on purpose. The doctor was alerted "
                       "immediately; speed matters more than a differential here.")
        else:
            (st.success if v["status"] == "pass" else st.error if v["status"] == "fail" else st.info)(
                f"Verifier: {v['status']} after {v['attempts']} attempt(s). {' '.join(v['notes'])}")
        if v.get("suggestions"):
            st.caption("Reviewer notes (non-blocking): " + " · ".join(v["suggestions"]))

        st.subheader("Follow-up items")
        st.dataframe([{"ID": f["id"], "Action": f["action"], "Owner": f["owner"], "Due": f["due"],
                       "Status": f["status"], "Appointment": f["appointment"], "Site": f.get("site") or "",
                       "Rule": f["rule"]}
                      for f in case["followups"]], hide_index=True, width='stretch')

    open_items = [f for f in case["followups"] if f["status"] not in ("completed",)]
    if open_items:
        with st.form("result"):
            pick = st.selectbox("Result received for", [f"{f['id']} - {f['action']}" for f in open_items])
            note = st.text_input("Result note", "normal")
            if st.form_submit_button("Record result"):
                post("/results", {"patient_id": pid, "followup_id": pick.split(" - ")[0], "note": note}); st.rerun()

with right:
    rad = [m for m in case["messages"] if m["to"] == "radiologist"]
    if rad:
        st.subheader("Radiology alerts")
        for m in rad[-4:]:
            st.markdown(f"**CareLoop** · {m['at']}\n\n{m['text']}")
    st.subheader("Doctor's phone")
    for m in [m for m in case["messages"] if m["to"] in ("doctor", "covering_doctor")][-6:]:
        who = "Doctor" if m["direction"] == "in" else ("CareLoop -> covering" if m["to"] == "covering_doctor" else "CareLoop")
        st.markdown(f"**{who}** · {m['at']}\n\n{m['text']}")
    with st.form("doc"):
        t = st.text_input("Reply as doctor", "APPROVE")
        if st.form_submit_button("Send"):
            post("/reply/doctor", {"patient_id": pid, "text": t}); st.rerun()

    st.subheader("Patient's phone")
    for m in [m for m in case["messages"] if m["to"] == "patient"][-6:]:
        who = "Patient" if m["direction"] == "in" else "CareLoop"
        st.markdown(f"**{who}** · {m['at']}\n\n{m['text']}")
    with st.form("pt"):
        t = st.text_input("Reply as patient", "OK")
        if st.form_submit_button("Send"):
            post("/reply/patient", {"patient_id": pid, "text": t}); st.rerun()

with st.expander("Audit log"):
    for e in reversed(case["audit_log"]):
        st.text(f"{e['at']}  [{e['agent']}]  {e['action']}")
