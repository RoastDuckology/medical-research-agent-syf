"""Shared state definition for the LangGraph workflow."""

from typing import TypedDict


class RetrievedDocument(TypedDict, total=False):
    """A document chunk retrieved from the local knowledge base."""

    content: str
    source: str
    filename: str
    date: str
    score: float


class ResearchState(TypedDict):
    """State passed between workflow nodes."""

    query: str
    memory: str
    retrieved_documents: list[RetrievedDocument]
    research_results: list[dict]
    report: str
