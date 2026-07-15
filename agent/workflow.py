"""High-level workflow entry point."""

from agent.graph import build_graph
from agent.state import ResearchState


def invoke(state: ResearchState) -> ResearchState:
    """Run the minimal LangGraph workflow with the provided state."""

    graph = build_graph()
    return graph.invoke(state)
