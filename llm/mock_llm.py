"""Mock LLM provider for local development without API credentials."""

from __future__ import annotations

import re
from typing import Any

from llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Deterministic fallback implementation used for local smoke tests."""

    def generate(self, prompt: str) -> str:
        """Return a conservative fallback report when only a prompt is available."""

        query = _extract_between(prompt, "用户问题：", "\n\n") or "当前问题"
        return (
            "# LLM Research Report (local fallback)\n\n"
            "生成器：MockLLM\n\n"
            f"当前问题：{query.strip()}\n\n"
            "由于未连接到可用的真实 LLM，本回答只能说明流程已运行。"
            "需要进一步核查检索片段、原始链接和 PDF 页码后再形成正式结论。"
        )

    def generate_from_state(self, state: dict[str, Any]) -> str:
        """Generate a source-aware fallback answer from structured agent state."""

        query = state.get("query", "")
        retrieved = state.get("retrieved_documents", []) or []
        results = state.get("research_results", []) or []

        lines = [
            "# LLM Research Report (local fallback)",
            "",
            "生成器：MockLLM",
            "",
            f"当前问题：{query}",
            "",
            "## 直接回答",
            "",
        ]

        evidence_lines = _evidence_summaries(retrieved)
        if evidence_lines:
            lines.append("根据当前 RAG 召回材料，可以先给出保守回答：")
            lines.extend(f"- {line}" for line in evidence_lines[:5])
        else:
            lines.append("当前没有召回足够的本地材料，信息不足以形成可靠回答。")

        if results:
            lines.extend(["", "## 外部工具结果", ""])
            for result in results[:5]:
                title = result.get("title") or result.get("name") or result.get("type", "result")
                url = result.get("url") or result.get("html_url") or result.get("output_path", "")
                lines.append(f"- {title}: {url}".rstrip(": "))

        lines.extend(
            [
                "",
                "## 信息不足与风险",
                "",
                "- 这是 MockLLM fallback 生成的回答，表示真实 LLM 调用失败后的本地降级结果。",
                "- 需要进一步核查原始 PDF、arXiv 链接、实验记录和工具结果后，再作为正式科研结论使用。",
            ]
        )
        return "\n".join(lines)


def _evidence_summaries(documents: list[dict[str, Any]]) -> list[str]:
    summaries: list[str] = []
    for document in documents:
        filename = document.get("filename") or document.get("source", "unknown source")
        page = document.get("page_start")
        abs_url = document.get("abs_url") or document.get("pdf_url") or ""
        content = _compact(document.get("content", ""), 260)
        locator = f"{filename} p.{page}" if page else str(filename)
        if abs_url:
            locator = f"{locator} ({abs_url})"
        summaries.append(f"{content} 来源：{locator}")
    return summaries


def _compact(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _extract_between(text: str, start: str, end: str) -> str:
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else ""
