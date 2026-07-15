"""LLM-based report node.

The node keeps the existing LangGraph workflow unchanged. It builds a prompt
from retrieved memory and research tool results, then delegates text generation
to the configured LLM provider.
"""

import json

from agent.state import ResearchState
from llm.provider import get_llm


def _build_report_prompt(state: ResearchState) -> str:
    """Build a structured report-generation prompt from workflow state."""

    return "\n\n".join(
        [
            "Generate a structured Markdown research report based on the information below.",
            "The report should include: research question, personal research memory, external paper/code signals, integrated analysis, and next-step suggestions.",
            "Do not invent unsupported facts. If information is insufficient, state that clearly.",
            f"User query: {state['query']}",
            f"Personal research memory:\n{state.get('memory', '')}",
            "Structured external research results:",
            json.dumps(state.get("research_results", []), ensure_ascii=False, indent=2),
        ]
    )


def report_node(state: ResearchState) -> dict[str, str]:
    """Generate a research report with the configured LLM provider."""

    prompt = _build_report_prompt(state)
    llm = get_llm()
    report = llm.generate(prompt)
    return {"report": report}

