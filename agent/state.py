"""Shared state definition for the LangGraph workflow."""

from typing import TypedDict


class RetrievedDocument(TypedDict, total=False):
    """A document chunk retrieved from the local knowledge base."""

    content: str
    source: str
    filename: str
    date: str
    document_type: str
    section_title: str
    heading_path: str
    chunk_index: int
    chunk_preview: str
    score: float
    memory_score: float
    dense_score: float
    keyword_score: float
    importance: float


class ResearchState(TypedDict, total=False):
    """State passed between workflow nodes."""

    query: str
    task_type: str
    plan: list[dict]
    selected_tools: list[str]
    reasoning_trace: list[dict]
    memory: str
    memory_records: list[dict]
    retrieved_documents: list[RetrievedDocument]
    research_results: list[dict]
    report: str
    report_error: str
    reflection: dict
    progress: list[dict]
    context_summary: dict
