from langgraph.graph import StateGraph, END
from src.state_planning import PlanningState
from src.nodes.planning_nodes import (
    search_policy_node,
    gather_intel_node,
    drafter_node,
    reviewer_academic_node,
    reviewer_emotional_node,
    consensus_check_node,
    plan_output_node,
    revision_node
)

def should_revise_or_output(state: PlanningState) -> str:
    iteration = state.get("iteration_count", 0)
    needs_revision = state.get("needs_revision", False)

    if iteration >= 3:
        return "output"

    if needs_revision:
        return "revise"

    return "output"

workflow = StateGraph(PlanningState)

workflow.add_node("search_policy", search_policy_node)
workflow.add_node("gather_intel", gather_intel_node)
workflow.add_node("drafter", drafter_node)
workflow.add_node("reviewer_academic", reviewer_academic_node)
workflow.add_node("reviewer_emotional", reviewer_emotional_node)
workflow.add_node("consensus_check", consensus_check_node)
workflow.add_node("revise", revision_node)
workflow.add_node("plan_output", plan_output_node)

workflow.set_entry_point("search_policy")

workflow.add_edge("search_policy", "gather_intel")
workflow.add_edge("gather_intel", "drafter")
workflow.add_edge("drafter", "reviewer_academic")
workflow.add_edge("reviewer_academic", "reviewer_emotional")
workflow.add_edge("reviewer_emotional", "consensus_check")

workflow.add_conditional_edges(
    "consensus_check",
    should_revise_or_output,
    {
        "revise": "revise",
        "output": "plan_output"
    }
)

workflow.add_edge("revise", "reviewer_academic")
workflow.add_edge("plan_output", END)

planning_app = workflow.compile()