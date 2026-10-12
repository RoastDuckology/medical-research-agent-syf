"""Concise Chinese Markdown formatting for daily AI briefs."""

from __future__ import annotations

from datetime import date

from brief.state import BriefItem
from brief.topics import TOP_VENUE_HINTS


def format_daily_brief(
    *,
    brief_date: str,
    days: int,
    topics: list[str],
    papers: list[BriefItem],
    repositories: list[BriefItem],
) -> str:
    """Format a concise Chinese daily brief with original source links."""

    lines = [
        f"# 每日 AI 科研简报 | {brief_date}",
        "",
        f"近 {days} 天追踪主题：{', '.join(topics)}。",
        f"生成日期：{date.today().isoformat()}。",
        "筛选原则：优先保留近期、主题相关、信息完整且带原始链接的论文和项目。",
        "",
        "## 论文精选",
        "",
    ]

    if papers:
        for index, item in enumerate(papers, start=1):
            lines.extend(_format_paper(index, item))
    else:
        lines.extend(["近期期限内未筛选出带原始链接的高相关 arXiv 论文。", ""])

    lines.extend(["## GitHub 项目精选", ""])
    if repositories:
        for index, item in enumerate(repositories, start=1):
            lines.extend(_format_repository(index, item))
    else:
        lines.extend(["近期期限内未筛选出带原始链接的高相关 GitHub 项目。", ""])

    lines.extend(
        [
            "## 顶刊顶会线索",
            "",
            _format_venue_note(papers),
            "",
            "## 今日判断",
            "",
            "以上内容是信息筛选结果，不等同于已经验证的科研结论。arXiv 论文需要继续检查实验设置、代码可复现性和是否经过同行评审；GitHub 项目需要继续查看 README、许可证、issue 和最近提交。后续深度调研应从每条信息的原始链接进入。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_paper(index: int, item: BriefItem) -> list[str]:
    title = item.get("title", "Untitled")
    summary = item.get("summary_cn") or _fallback_paper_summary(item)

    return [
        f"### {index}. [{title}]({item['url']})",
        "",
        summary,
        "",
    ]


def _format_repository(index: int, item: BriefItem) -> list[str]:
    title = item.get("title", "Untitled")
    summary = item.get("summary_cn") or _fallback_repository_summary(item)

    return [
        f"### {index}. [{title}]({item['url']})",
        "",
        summary,
        "",
    ]


def _format_venue_note(papers: list[BriefItem]) -> str:
    hints = []
    for item in papers:
        text = " ".join(
            [
                item.get("title", ""),
                item.get("summary", ""),
                item.get("venue_hint", ""),
            ]
        ).lower()
        matched = [venue for venue in TOP_VENUE_HINTS if venue.lower() in text]
        if matched:
            hints.append(f"- [{item.get('title', 'Untitled')}]({item['url']})：疑似包含 {', '.join(matched)} 线索。")

    if hints:
        return "\n".join(hints)
    return "本次筛选结果未从 arXiv 元数据中稳定识别到顶刊顶会名称。是否为顶刊顶会论文，需要进入原始链接、作者主页或会议官网进一步确认。"


def _fallback_paper_summary(item: BriefItem) -> str:
    return (
        f"这是一篇与“{item.get('topic', '未标注主题')}”相关的论文，发布时间为"
        f" {item.get('published', '未知时间')}。作者包括 {_authors(item)}，来源线索为"
        f" {item.get('venue_hint') or 'arXiv 预印本'}。原始摘要较长，建议进入原文核查方法、实验和局限性。"
        f"原始链接：{item.get('url', '')}"
    )


def _fallback_repository_summary(item: BriefItem) -> str:
    return (
        f"这是一个与“{item.get('topic', '未标注主题')}”相关的 GitHub 项目，最近更新为"
        f" {item.get('updated_at', '未知时间')}。项目语言为 {item.get('language') or '未标注'}，"
        f"Stars 为 {item.get('stars', 0)}。复用前需要查看 README、许可证、issue 和最近提交。"
        f"原始链接：{item.get('url', '')}"
    )


def _authors(item: BriefItem) -> str:
    authors = item.get("authors", [])
    if not authors:
        return "未提供作者"
    display = ", ".join(authors[:3])
    if len(authors) > 3:
        display += " 等"
    return display


def _shorten(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."
