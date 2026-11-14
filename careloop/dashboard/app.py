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
    return httpx.get(API + path, timeout=60).json()


def post(path, body=None):
    r = httpx.post(API + path, json=body or {}, timeout=120)
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
    demo_pid = st.selectbox("Load next demo scan for", ["P-003", "P-002", "P-001", "P-004"])
    if st.button("Load scan"):
        post(f"/demo/load/{demo_pid}")
        st.rerun()
    c1, c2, c3 = st.columns(3)
    if c1.button("+15 min"):
        post("/clock/advance", {"minutes": 15}); st.rerun()
    if c2.button("+1 day"):
        post("/clock/advance", {"days": 1}); st.rerun()
    if c3.button("+30 days"):
        post("/clock/advance", {"days": 30}); st.rerun()
    if st.button("Reset demo"):
        post("/reset"); st.rerun()

if not pid:
    st.info("No cases yet. Load a demo scan from the sidebar.")
    st.stop()

case = get(f"/cases/{pid}")
p = case["patient"]
st.header(f"{URG_COLOR[case['urgency']]} {p['name']} ({p['patient_id']}), {p['age']}{p['sex']}")
st.caption(f"State: **{case['state']}** | Ordering: {p['ordering_doctor']} | History: {'; '.join(p['history'])}")

left, right = st.columns([3, 2])

with left:
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
    (st.success if v["status"] == "pass" else st.error if v["status"] == "fail" else st.info)(
        f"Verifier: {v['status']} after {v['attempts']} attempt(s). {' '.join(v['notes'])}")

    st.subheader("Follow-up items")
    st.dataframe([{"ID": f["id"], "Action": f["action"], "Owner": f["owner"], "Due": f["due"],
                   "Status": f["status"], "Appointment": f["appointment"], "Rule": f["rule"]}
                  for f in case["followups"]], hide_index=True, width='stretch')

with right:
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
