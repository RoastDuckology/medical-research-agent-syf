"""LangGraph graph construction for the research assistant workflow."""

from langgraph.graph import END, START, StateGraph

from agent.nodes.daily_brief_node import daily_brief_node
from agent.nodes.memory_node import memory_node
from agent.nodes.planner_node import planner_node
from agent.nodes.reflection_node import reflection_node
from agent.nodes.research_node import research_node
from agent.nodes.report_node import report_node
from agent.state import ResearchState


def build_graph():
    """Build and compile the minimal research assistant workflow."""

    graph = StateGraph(ResearchState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("daily_brief_node", daily_brief_node)
    graph.add_node("memory_node", memory_node)
    graph.add_node("research_node", research_node)
    graph.add_node("report_node", report_node)
    graph.add_node("reflection_node", reflection_node)

    graph.add_edge(START, "planner_node")
    graph.add_conditional_edges(
        "planner_node",
        _route_after_planner,
        {
            "daily_brief_node": "daily_brief_node",
            "memory_node": "memory_node",
        },
    )
    graph.add_edge("daily_brief_node", "reflection_node")
    graph.add_edge("memory_node", "research_node")
    graph.add_edge("research_node", "report_node")
    graph.add_edge("report_node", "reflection_node")
    graph.add_edge("reflection_node", END)

    return graph.compile()


def _route_after_planner(state: ResearchState) -> str:
    """Route daily brief requests away from the general research path."""

    if state.get("task_type") == "daily_brief":
        return "daily_brief_node"
    return "memory_node"
