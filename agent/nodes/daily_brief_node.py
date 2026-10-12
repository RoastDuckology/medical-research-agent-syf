"""LangGraph node for standalone DailyBriefAgent execution."""

from __future__ import annotations

from agent.state import ResearchState
from tools.daily_brief_tool import DailyBriefTool


def daily_brief_node(state: ResearchState) -> dict:
    """Run DailyBriefAgent and return a concise source-linked report."""

    query = state.get("query", "")
    result = DailyBriefTool().run(query=query)
    report = _format_daily_brief_result(result)
    return {
        "research_results": result,
        "reasoning_trace": [
            {
                "thought": "The planner classified this request as a daily brief task.",
                "action": "daily_brief",
                "observation": f"{len(result)} daily brief result(s) returned.",
            }
        ],
        "report": report,
    }


def _format_daily_brief_result(results: list[dict]) -> str:
    if not results:
        return (
            "# 每日 AI 简报\n\n"
            "DailyBriefAgent 未返回结果。需要检查外部数据源、网络或配置。"
        )

    item = results[0]
    output_path = item.get("output_path", "")
    topics = ", ".join(item.get("topics", []))
    return "\n".join(
        [
            "# 每日 AI 简报",
            "",
            f"- 日期：{item.get('date', '')}",
            f"- 时间窗口：近 {item.get('days', '')} 天",
            f"- 主题：{topics}",
            f"- 论文条目数：{item.get('paper_count', 0)}",
            f"- GitHub 项目数：{item.get('repo_count', 0)}",
            f"- 输出文件：{output_path}",
            "",
            "简报正文已写入上述 Markdown 文件。所有入选论文和项目均应在文件中保留原始链接；如条目数为 0，需要检查 arXiv/GitHub 限流或查询主题。",
        ]
    )
