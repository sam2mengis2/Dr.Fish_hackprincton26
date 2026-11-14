"""Agent 9 - Clinician Comms (Photon texts; keyword parsing in code, LLM only for EDIT notes).

Outgoing: summary + top candidate + finding IDs + suggested actions + reply options.
Replies: APPROVE / EDIT <note> / REJECT <reason> / CALL. Anything else -> "Did you mean...?"
Timers: reminder, then escalate to the covering doctor (urgent: 15/30 min, routine: 2/4 h).
"""
from __future__ import annotations

import re
from datetime import timedelta

from ..runtime import Runtime
from ..schema import Approval, Case, CaseState, Message, Timer


def _initials(name: str) -> str:
    return "".join(p[0] for p in name.split() if p)[:3].upper() + "."


def compose(case: Case) -> str:
    p = case.patient
    head = f"CareLoop | Patient {_initials(p.name)} ({p.patient_id}) | {case.urgency.upper() if case.urgency else ''}"
    if case.state == CaseState.NEEDS_HUMAN_REVIEW:
        body = "AI reasoning failed verification twice. Please review the case on the dashboard."
    elif case.urgency == "critical":
        fs = ", ".join(f"{case.finding(i).finding} ({i})" for i in case.urgency_reasons if case.finding(i))
        body = f"CRITICAL finding: {fs}."
    else:
        fs = "; ".join(f"{f.finding} {f.location} ({f.finding_id})".replace("  ", " ")
                       for f in case.findings if f.finding_id in (case.differential[0].evidence_for if case.differential else []))
        top = case.differential[0].dx if case.differential else "insufficient information"
        body = f"Findings: {fs or 'see dashboard'}.\nTop possibility: {top}."
    if case.plan_summary == "No new follow-up needed":
        return f"{head} | FYI\n{body}\nNo new actions suggested; existing plan continues.\nReply APPROVE to acknowledge, or EDIT <note>."
    actions = f"\nSuggested: {case.plan_summary}." if case.plan_summary else ""
    return f"{head}\n{body}{actions}\nReply APPROVE, EDIT <note>, REJECT <reason>, or CALL."


def send_plan(case: Case, rt: Runtime) -> Case:
    text = compose(case)
    rt.messenger.send("doctor", case.patient.patient_id, text)
    case.messages.append(Message(to="doctor", text=text, at=rt.clock.iso()))
    t = rt.config["timers"]
    fast = case.urgency in ("critical", "urgent")
    rem = t["urgent_reminder_min"] if fast else t["routine_reminder_min"]
    esc = t["urgent_escalate_min"] if fast else t["routine_escalate_min"]
    now = rt.clock.now()
    case.timers = [x for x in case.timers if x.kind not in ("approval_reminder", "approval_escalate")]
    case.timers.append(Timer(kind="approval_reminder", fire_at=(now + timedelta(minutes=rem)).isoformat(timespec="minutes")))
    case.timers.append(Timer(kind="approval_escalate", fire_at=(now + timedelta(minutes=esc)).isoformat(timespec="minutes")))
    if case.state != CaseState.NEEDS_HUMAN_REVIEW:
        case.state = CaseState.AWAITING_APPROVAL
    case.log("clinician_comms", "plan texted to ordering doctor; approval timers set", rt.clock.iso())
    return case


REPLY = re.compile(r"^\s*(APPROVE|APPROVED|YES|EDIT|REJECT|CALL)\b[:\s-]*(.*)$", re.I | re.S)


def parse(text: str) -> tuple[str | None, str]:
    m = REPLY.match(text)
    if not m:
        return None, text.strip()
    word = m.group(1).upper()
    word = "APPROVE" if word in ("APPROVED", "YES") else word
    return word, m.group(2).strip()


def handle_reply(case: Case, text: str, rt: Runtime, by: str | None = None) -> tuple[Case, str]:
    """Returns (case, outcome). outcome in {APPROVE, EDIT, REJECT, CALL, UNCLEAR}."""
    by = by or case.patient.ordering_doctor
    case.messages.append(Message(to="doctor", direction="in", text=text, at=rt.clock.iso()))
    decision, note = parse(text)
    if decision is None:
        q = "Sorry, I didn't catch that. Reply APPROVE, EDIT <note>, REJECT <reason>, or CALL."
        rt.messenger.send("doctor", case.patient.patient_id, q)
        case.messages.append(Message(to="doctor", text=q, at=rt.clock.iso()))
        return case, "UNCLEAR"
    case.approvals.append(Approval(by=by, decision=decision, note=note, at=rt.clock.iso()))
    for t in case.timers:
        if t.kind in ("approval_reminder", "approval_escalate"):
            t.done = True
    if decision == "APPROVE":
        case.state = CaseState.APPROVED
    elif decision == "EDIT":
        case.state = CaseState.EDITED
        _apply_edit(case, note, rt)
    elif decision == "REJECT":
        case.state = CaseState.REJECTED
    else:
        case.state = CaseState.DOCTOR_CALLING
    case.log("clinician_comms", f"doctor replied {decision} {note!r}", rt.clock.iso())
    return case, decision


def _apply_edit(case: Case, note: str, rt: Runtime) -> None:
    """Minimal edit grammar for mock mode: 'add <test> by <YYYY-MM-DD>' / 'remove FU-n'.
    With a real LLM, swap this for an LLM call that returns a revised follow-up list."""
    m = re.match(r"add (.+?)(?: by (\d{4}-\d{2}-\d{2}))?$", note, re.I)
    if m:
        from .planner import _add

        due = m.group(2) or (rt.clock.now() + timedelta(days=7)).date().isoformat()
        _add(case, m.group(1), due, "Added by doctor (EDIT)", [])
    m = re.match(r"remove (FU-\d+)", note, re.I)
    if m:
        case.followups = [f for f in case.followups if f.id.upper() != m.group(1).upper()]
    case.plan_summary = "; ".join(f"{fu.action} by {fu.due}" for fu in case.followups if fu.status != "completed")
