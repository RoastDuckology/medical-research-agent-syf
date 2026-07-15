"""Research node that calls registered external research tools."""

from agent.state import ResearchState
from tools.registry import create_default_tool_registry


def research_node(state: ResearchState) -> dict[str, list[dict]]:
    """Run all registered research tools for the user query."""

    registry = create_default_tool_registry()
    return {"research_results": registry.run_all(query=state["query"])}
