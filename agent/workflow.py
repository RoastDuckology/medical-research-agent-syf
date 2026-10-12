"""High-level workflow entry point."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agent.graph import build_graph
from agent.state import ResearchState


def invoke(state: ResearchState) -> ResearchState:
    """Run the minimal LangGraph workflow with the provided state."""

    graph = build_graph()
    return graph.invoke(state)


def stream_updates(state: ResearchState) -> Iterator[tuple[str, dict[str, Any], ResearchState]]:
    """Stream LangGraph node updates while reconstructing the final state."""

    graph = build_graph()
    current_state: ResearchState = dict(state)
    for chunk in graph.stream(state, stream_mode="updates"):
        for node_name, update in chunk.items():
            if isinstance(update, dict):
                current_state.update(update)
                yield node_name, update, current_state
