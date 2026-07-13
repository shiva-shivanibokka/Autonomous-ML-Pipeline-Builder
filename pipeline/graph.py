"""
pipeline.graph — LangGraph StateGraph wiring for the ML Pipeline Builder.

Graph topology:
    START
      → orchestrator
      → data_analyst
      → feature_engineer
      → model_trainer
      → evaluator
      → code_generator
      → deployment_agent
      → END

Conditional edges short-circuit to END on any agent error,
setting state["status"] = "failed" and state["error"] = message.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agents.code_generator import run_code_generator
from agents.data_analyst import run_data_analyst
from agents.deployment_agent import run_deployment_agent
from agents.evaluator import run_evaluator
from agents.feature_engineer import run_feature_engineer
from agents.model_trainer import run_model_trainer
from agents.orchestrator import run_orchestrator
from agents.state import AgentState


def _route_or_error(next_node: str):
    """
    Return a conditional edge function that routes to `next_node`
    unless the state has an error, in which case it routes to END.
    """

    def _route(state: AgentState) -> str:
        if state.get("error"):
            return END
        return next_node

    return _route


def build_graph() -> StateGraph:
    """
    Construct and compile the full ML Pipeline LangGraph StateGraph.

    Returns the compiled graph ready for .invoke() / .stream().
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    graph.add_node("orchestrator", run_orchestrator)
    graph.add_node("data_analyst", run_data_analyst)
    graph.add_node("feature_engineer", run_feature_engineer)
    graph.add_node("model_trainer", run_model_trainer)
    graph.add_node("evaluator", run_evaluator)
    graph.add_node("code_generator", run_code_generator)
    graph.add_node("deployment_agent", run_deployment_agent)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "orchestrator")

    # ── Conditional edges (error short-circuit) ───────────────────────────────
    graph.add_conditional_edges(
        "orchestrator",
        _route_or_error("data_analyst"),
        {"data_analyst": "data_analyst", END: END},
    )
    graph.add_conditional_edges(
        "data_analyst",
        _route_or_error("feature_engineer"),
        {"feature_engineer": "feature_engineer", END: END},
    )
    graph.add_conditional_edges(
        "feature_engineer",
        _route_or_error("model_trainer"),
        {"model_trainer": "model_trainer", END: END},
    )
    graph.add_conditional_edges(
        "model_trainer",
        _route_or_error("evaluator"),
        {"evaluator": "evaluator", END: END},
    )
    graph.add_conditional_edges(
        "evaluator",
        _route_or_error("code_generator"),
        {"code_generator": "code_generator", END: END},
    )
    graph.add_conditional_edges(
        "code_generator",
        _route_or_error("deployment_agent"),
        {"deployment_agent": "deployment_agent", END: END},
    )
    graph.add_edge("deployment_agent", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Return the cached compiled graph singleton."""
    return build_graph()
