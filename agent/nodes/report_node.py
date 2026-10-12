"""LLM-based report node with deterministic source-aware post-processing."""

from __future__ import annotations

from typing import Any

from agent.state import ResearchState
from config.settings import get_settings
from context.builder import ContextBuilder
from llm.mock_llm import MockLLM
from llm.provider import get_llm


def _build_report_prompt(state: ResearchState) -> str:
    """Build a structured report-generation prompt from workflow state."""

    built_context = ContextBuilder(
        budget_tokens=get_settings().context_budget_tokens,
    ).build(state)
    return built_context.prompt


def report_node(state: ResearchState) -> dict[str, Any]:
    """Generate a research report with the configured LLM provider."""

    built_context = ContextBuilder(
        budget_tokens=get_settings().context_budget_tokens,
    ).build(state)
    prompt = built_context.prompt
    llm = get_llm()
    try:
        report = llm.generate(prompt)
        return {
            "report": _append_source_appendix(report, state),
            "context_summary": built_context.summary(),
        }
    except Exception as exc:
        fallback_report = MockLLM().generate_from_state(state)
        return {
            "report": _append_source_appendix(fallback_report, state),
            "report_error": _compact_error(str(exc)),
            "context_summary": built_context.summary(),
        }


def _append_source_appendix(report: str, state: ResearchState) -> str:
    """Append machine-checkable sources so reflection and QA metrics are stable."""

    lines = [report.rstrip()]
    source_lines = _source_lines(state)
    if source_lines:
        lines.extend(["", "### Sources Used", ""])
        lines.extend(source_lines)
    return "\n".join(lines).strip() + "\n"


def _source_lines(state: ResearchState) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for document in state.get("retrieved_documents", []) or []:
        line = _document_source_line(document)
        if line and line not in seen:
            lines.append(line)
            seen.add(line)
        if len(lines) >= 6:
            break

    for result in state.get("research_results", []) or []:
        line = _result_source_line(result)
        if line and line not in seen:
            lines.append(line)
            seen.add(line)
        if len(lines) >= 10:
            break
    return lines


def _document_source_line(document: dict[str, Any]) -> str:
    filename = document.get("filename") or document.get("source")
    if not filename:
        return ""
    parts = [f"- file: {filename}"]
    if document.get("document_type"):
        parts.append(f"type: {document['document_type']}")
    if document.get("page_start"):
        page = document["page_start"]
        page_end = document.get("page_end")
        parts.append(f"page: {page}-{page_end}" if page_end and page_end != page else f"page: {page}")
    if document.get("heading_path"):
        parts.append(f"heading_path: {document['heading_path']}")
    if document.get("abs_url"):
        parts.append(f"Source: {document['abs_url']}")
    elif document.get("pdf_url"):
        parts.append(f"Source: {document['pdf_url']}")
    return "; ".join(parts)


def _result_source_line(result: dict[str, Any]) -> str:
    url = result.get("url") or result.get("html_url") or result.get("output_path")
    if not url:
        return ""
    title = result.get("title") or result.get("name") or result.get("type", "external_result")
    return f"- external: {title}; Source: {url}"


def _compact_error(error: str, max_chars: int = 360) -> str:
    normalized = " ".join(error.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."
