"""Concise Chinese summarization for selected daily brief items."""

from __future__ import annotations

from config.settings import get_settings
from llm.provider import get_llm


def summarize_selected_items(items: list[dict], *, item_type: str) -> list[dict]:
    """Attach concise Chinese summaries to selected items when an LLM is available."""

    if not get_settings().deepseek_api_key:
        return [_with_fallback_summary(item, item_type) for item in items]

    summarized: list[dict] = []
    llm = get_llm()
    for item in items:
        try:
            summary_cn = llm.generate(_build_summary_prompt(item, item_type)).strip()
        except Exception:
            summarized.append(_with_fallback_summary(item, item_type))
            continue

        normalized = _normalize_summary(summary_cn)
        summarized.append({**item, "summary_cn": normalized})
    return summarized


def _build_summary_prompt(item: dict, item_type: str) -> str:
    if item_type == "paper":
        return "\n".join(
            [
                "请用中文总结下面这篇论文，控制在3-5句话。",
                "不要编造未提供的信息，不要删除原始链接。",
                "总结应包括：研究主题、可能贡献、可信度提醒、原始链接。",
                f"标题：{item.get('title', '')}",
                f"作者：{', '.join(item.get('authors', [])[:5])}",
                f"发布时间：{item.get('published', '')}",
                f"来源/venue线索：{item.get('venue_hint', '') or 'arXiv 预印本'}",
                f"摘要：{item.get('summary', '')}",
                f"原始链接：{item.get('url', '')}",
            ]
        )

    return "\n".join(
        [
            "请用中文总结下面这个GitHub项目，控制在3-5句话。",
            "不要编造未提供的信息，不要删除原始链接。",
            "总结应包括：项目用途、活跃度线索、使用前需要核查的点、原始链接。",
            f"仓库：{item.get('title', '')}",
            f"描述：{item.get('summary', '')}",
            f"语言：{item.get('language', '')}",
            f"Stars：{item.get('stars', 0)}",
            f"最近更新：{item.get('updated_at', '')}",
            f"原始链接：{item.get('url', '')}",
        ]
    )


def _with_fallback_summary(item: dict, item_type: str) -> dict:
    if item_type == "paper":
        summary = (
            f"这是一篇与“{item.get('topic', '未标注主题')}”相关的论文，发布时间为"
            f" {item.get('published', '未知时间')}。"
            f"作者包括 {_authors(item)}，来源线索为 {item.get('venue_hint') or 'arXiv 预印本'}。"
            "基于当前元数据只能做初步判断，具体贡献、实验设置和局限性需要进入原文核查。"
            f"原始链接：{item.get('url', '')}"
        )
    else:
        summary = (
            f"这是一个与“{item.get('topic', '未标注主题')}”相关的 GitHub 项目，最近更新为"
            f" {item.get('updated_at', '未知时间')}。"
            f"项目语言为 {item.get('language') or '未标注'}，Stars 为 {item.get('stars', 0)}。"
            "基于当前元数据只能做初步判断，复用前需要查看 README、许可证和最近提交。"
            f"原始链接：{item.get('url', '')}"
        )
    return {**item, "summary_cn": summary}


def _normalize_summary(text: str) -> str:
    lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    normalized = " ".join(lines)
    return _ensure_terminal_punctuation(normalized)


def _ensure_terminal_punctuation(text: str) -> str:
    if not text:
        return text
    if text[-1] in "。！？.!?）)】]":
        return text
    return text + "。"


def _authors(item: dict) -> str:
    authors = item.get("authors", [])
    if not authors:
        return "未提供作者"
    display = "、".join(authors[:3])
    if len(authors) > 3:
        display += " 等"
    return display
