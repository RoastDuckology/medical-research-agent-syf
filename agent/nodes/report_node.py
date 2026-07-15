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
            "请基于以下信息生成一份中文 Markdown 研究报告。",
            "报告需要包含：研究问题、个人工作记录、外部论文/代码线索、综合分析、下一步建议。",
            "不要编造未提供的信息；如果信息不足，请明确说明。",
            f"用户问题：{state['query']}",
            f"个人工作记录：\n{state.get('memory', '')}",
            "结构化外部研究结果：",
            json.dumps(state.get("research_results", []), ensure_ascii=False, indent=2),
        ]
    )


def report_node(state: ResearchState) -> dict[str, str]:
    """Generate a research report with the configured LLM provider."""

    prompt = _build_report_prompt(state)
    llm = get_llm()
    report = llm.generate(prompt)
    return {"report": report}
