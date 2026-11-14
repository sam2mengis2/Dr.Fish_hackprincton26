"""Agent 10 - Patient Comms (LLM with strict limits). Runs ONLY after doctor approval.

Tiers: serious -> no diagnosis by text ("your doctor will call"); routine -> plain-language
explanation + booked appointment. Code guard blocks serious words in every patient text.
The LLM never sees the raw differential, only the doctor-approved plan.
"""
from __future__ import annotations

import re

from ..knowledge import SERIOUS_WORDS, patient_label
from ..llm import maybe
from ..runtime import Runtime
from ..schema import Case, Message

SYSTEM = """Write a short, warm text message (max 60 words, about grade-6 reading level) to a patient.
You may explain the follow-up plan and appointment below. Do NOT mention any diagnosis, disease
names, cancer, or anything alarming. Do not give medical advice. Sign off as 'your care team'.
Copy every appointment date and time EXACTLY as given (YYYY-MM-DD at HH:MM)."""

CLINICAL_Q = re.compile(r"\b(cancer|serious|dying|die|tumou?r|stop|medication|meds|dose|pain|worse|what is it|is it)\b", re.I)


def _first_name(case: Case) -> str:
    return case.patient.name.split()[0]


def _safe(text: str, fallback: str) -> str:
    return fallback if SERIOUS_WORDS.search(text) else text


def notify(case: Case, rt: Runtime) -> Case:
    doc = case.patient.ordering_doctor
    if case.message_tier == "serious":
        text = f"Hi {_first_name(case)}, {doc} reviewed your recent scans and will call you today to talk about next steps. - your care team"
    else:
        items = [fu for fu in case.followups if fu.status in ("open", "booked")]
        plan = "; ".join(f"{patient_label(fu.action)} on {fu.appointment.replace('T', ' at ') if fu.appointment else 'a date we will confirm'}"
                         for fu in items) or "no further tests"
        fallback = (f"Hi {_first_name(case)}, {doc} reviewed your scan. A routine follow-up is recommended: {plan}. "
                    f"Reply OK to confirm or RESCHEDULE. - your care team")
        if not items:  # nothing planned: don't let the LLM improvise reassurance
            fallback = (f"Hi {_first_name(case)}, {doc} reviewed your scan. Your care team will contact you "
                        f"about any next steps. - your care team")
        if maybe(rt.llm) and items:
            draft = rt.llm.text("comms", SYSTEM, f"Patient first name: {_first_name(case)}. Doctor: {doc}. Plan: {plan}. "
                                                  "End with: Reply OK to confirm or RESCHEDULE.")
            # Must keep every booked date; otherwise use the template (it always states them).
            dates = [fu.appointment[:10] for fu in items if fu.appointment]
            looks_raw = (not draft) or draft.lstrip().startswith(("[", "{")) or "signature" in draft
            text = fallback if looks_raw or not all(d in draft for d in dates) else _safe(draft, fallback)
        else:
            text = fallback
    text = _safe(text, f"Hi {_first_name(case)}, {doc} will contact you about your recent scan. - your care team")
    rt.messenger.send("patient", case.patient.patient_id, text)
    case.messages.append(Message(to="patient", tier=case.message_tier, text=text, at=rt.clock.iso()))
    for fu in case.followups:
        if fu.status == "open":
            fu.status = "patient_notified" if not fu.appointment else fu.status
    case.log("patient_comms", f"{case.message_tier} message sent", rt.clock.iso())
    return case


def handle_reply(case: Case, text: str, rt: Runtime) -> tuple[Case, str]:
    case.messages.append(Message(to="patient", direction="in", text=text, at=rt.clock.iso()))
    t = text.strip().upper()
    if t.startswith("OK") or t.startswith("YES"):
        outcome = "CONFIRMED"
    elif t.startswith("RESCHEDULE"):
        from .scheduler import reschedule_next

        reschedule_next(case, rt)
        outcome = "RESCHEDULED"
    elif CLINICAL_Q.search(text) or "?" in text:
        msg = "Thanks for asking. I've passed your question to your care team and they'll get back to you. - your care team"
        rt.messenger.send("patient", case.patient.patient_id, msg)
        rt.messenger.send("nurse", case.patient.patient_id, f"Patient question needs a clinician: {text!r}")
        case.messages.append(Message(to="patient", text=msg, at=rt.clock.iso()))
        case.messages.append(Message(to="nurse", text=f"Patient question: {text}", at=rt.clock.iso()))
        outcome = "HANDED_TO_CARE_TEAM"
    else:
        outcome = "NOTED"
    case.log("patient_comms", f"patient replied -> {outcome}", rt.clock.iso())
    return case, outcome
