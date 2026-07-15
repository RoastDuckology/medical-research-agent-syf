"""LangGraph graph construction for the research assistant workflow."""

from langgraph.graph import END, START, StateGraph

from agent.nodes.memory_node import memory_node
from agent.nodes.research_node import research_node
from agent.nodes.report_node import report_node
from agent.state import ResearchState


def build_graph():
    """Build and compile the minimal research assistant workflow."""

    graph = StateGraph(ResearchState)
    graph.add_node("memory_node", memory_node)
    graph.add_node("research_node", research_node)
    graph.add_node("report_node", report_node)

    graph.add_edge(START, "memory_node")
    graph.add_edge("memory_node", "research_node")
    graph.add_edge("research_node", "report_node")
    graph.add_edge("report_node", END)

    return graph.compile()
