"""Agent 0 - Supervisor: the LangGraph StateGraph that routes each case by its state.

  intake (done before invoke) -> perception -> extractor -> reconciler -> triage
     triage: critical ---------------------------------------------> planner (fast path)
     triage: otherwise -> reasoner -> verifier
        verifier: pass -> planner ; fail & retries left -> reasoner ; exhausted -> human review
  planner -> clinician_comms (texts the doctor; graph ends at AWAITING_APPROVAL)

The doctor's reply resumes the case through service.doctor_reply().
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import clinician_comms, extractor, perception, planner, reasoner, reconciler, triage, verifier
from .runtime import Runtime
from .schema import Case, CaseState


class GraphState(TypedDict):
    case: Case


def pick_reasoner(rt: Runtime):
    """llm.reasoner_backend: auto (mock rules or single LLM call, per llm.provider) | deepagent."""
    if rt.config["llm"].get("reasoner_backend", "auto") == "deepagent":
        from .agents import deep_reasoner

        return deep_reasoner.run
    return reasoner.run


def build_graph(rt: Runtime):
    def node(fn):
        def _run(state: GraphState) -> GraphState:
            return {"case": fn(state["case"], rt)}
        return _run

    def after_triage(state: GraphState) -> str:
        return "planner" if state["case"].urgency == "critical" else "reasoner"

    def after_reasoner(state: GraphState) -> str:
        return "verifier" if rt.config["pipeline"]["verifier_enabled"] else "planner"

    def after_verifier(state: GraphState) -> str:
        s = state["case"].state
        if s == CaseState.VERIFIED:
            return "planner"
        if s == CaseState.VERIFY_FAILED:
            return "reasoner"
        return "clinician_comms"  # NEEDS_HUMAN_REVIEW: flag it to the doctor

    g = StateGraph(GraphState)
    g.add_node("perception", node(perception.run))
    g.add_node("extractor", node(extractor.run))
    g.add_node("reconciler", node(reconciler.run))
    g.add_node("triage", node(triage.run))
    g.add_node("reasoner", node(pick_reasoner(rt)))
    g.add_node("verifier", node(verifier.run))
    g.add_node("planner", node(planner.run))
    g.add_node("clinician_comms", node(clinician_comms.send_plan))

    g.add_edge(START, "perception")
    g.add_edge("perception", "extractor")
    g.add_edge("extractor", "reconciler")
    g.add_edge("reconciler", "triage")
    g.add_conditional_edges("triage", after_triage, ["planner", "reasoner"])
    g.add_conditional_edges("reasoner", after_reasoner, ["verifier", "planner"])
    g.add_conditional_edges("verifier", after_verifier, ["planner", "reasoner", "clinician_comms"])
    g.add_edge("planner", "clinician_comms")
    g.add_edge("clinician_comms", END)
    return g.compile()
