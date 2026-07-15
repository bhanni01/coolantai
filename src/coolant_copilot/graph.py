"""LangGraph StateGraph wiring the nodes over GraphState.

                       ┌→ property_estimator ─┐
    START → research → generator              ├→ critic ─→ experiment_planner → END
                       └→ compliance_checker ─┘   │
                          ▲                       │
                          └───── revise loop ─────┘

The generator fans out to property_estimator and compliance_checker, which
run concurrently (same superstep) and write disjoint state channels
(property_estimates vs compliance_flags), so no reducer is needed; the
joined edge into the critic waits for both.

The critic's conditional edge proceeds to the planner when the top-scoring
candidate is accepted, and loops back to the generator otherwise — capped by
revision_count, which the generator increments (edge functions cannot write
state, and the critic incrementing would over-count on the planner path).
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from coolant_copilot.nodes.compliance_checker import make_compliance_checker_node
from coolant_copilot.nodes.critic import make_critic_node
from coolant_copilot.nodes.experiment_planner import make_experiment_planner_node
from coolant_copilot.nodes.generator import make_generator_node
from coolant_copilot.nodes.property_estimator import make_property_estimator_node
from coolant_copilot.nodes.research import make_research_node
from coolant_copilot.schemas.extraction import ExtractedFluidProfile
from coolant_copilot.state import GraphState

MAX_REVISIONS = 3


def route_after_critic(state: GraphState) -> Literal["generator", "experiment_planner"]:
    if not state.critic_scores:
        # Nothing scorable was ever produced; revising won't help.
        return "experiment_planner"
    top = max(state.critic_scores, key=lambda s: s.overall)
    if top.verdict == "accept":
        return "experiment_planner"
    if state.revision_count < MAX_REVISIONS:
        return "generator"
    return "experiment_planner"


def build_graph(
    vectorstore: VectorStore,
    *,
    research_llm: BaseChatModel | None = None,
    generator_llm: BaseChatModel | None = None,
    estimator_llm: BaseChatModel | None = None,
    compliance_llm: BaseChatModel | None = None,
    critic_llm: BaseChatModel | None = None,
    planner_llm: BaseChatModel | None = None,
    reference_profiles: list[ExtractedFluidProfile] | None = None,
    research_score_threshold: float | None = None,
):
    graph = StateGraph(GraphState)
    graph.add_node(
        "research",
        make_research_node(
            vectorstore,
            research_llm,
            reference_profiles,
            score_threshold=research_score_threshold,
        ),
    )
    graph.add_node("generator", make_generator_node(generator_llm))
    graph.add_node(
        "property_estimator", make_property_estimator_node(reference_profiles, estimator_llm)
    )
    graph.add_node("compliance_checker", make_compliance_checker_node(compliance_llm))
    graph.add_node("critic", make_critic_node(critic_llm))
    graph.add_node("experiment_planner", make_experiment_planner_node(planner_llm))

    graph.add_edge(START, "research")
    graph.add_edge("research", "generator")
    # Fan-out: both evaluation nodes run concurrently after the generator.
    graph.add_edge("generator", "property_estimator")
    graph.add_edge("generator", "compliance_checker")
    # Join: the critic waits for both branches.
    graph.add_edge(["property_estimator", "compliance_checker"], "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"generator": "generator", "experiment_planner": "experiment_planner"},
    )
    graph.add_edge("experiment_planner", END)
    return graph.compile()
