"""Tool wrapper for the standalone daily brief agent."""

from __future__ import annotations

import os
from typing import Any

from brief.workflow import invoke_daily_brief
from tools.base import BaseTool


class DailyBriefTool(BaseTool):
    """Generate a recent AI research daily brief."""

    name = "daily_brief"
    description = "Generate a Markdown daily AI research brief with original source links."

    def run(self, **kwargs: Any) -> list[dict]:
        """Generate a daily brief and return its metadata."""

        if os.getenv("RESEARCH_TOOL_MODE", "").strip().lower() == "recorded":
            return [
                {
                    "type": "daily_brief",
                    "source": "daily_brief_agent",
                    "recorded": True,
                    "date": str(kwargs.get("date") or "2026-08-04"),
                    "days": int(kwargs.get("days", 7)),
                    "topics": kwargs.get("topics") or [str(kwargs.get("query", "AI research brief"))],
                    "paper_count": 2,
                    "repo_count": 2,
                    "output_path": "data/recordings/daily_brief_recorded.md",
                    "url": "data/recordings/daily_brief_recorded.md",
                }
            ]

        topics = kwargs.get("topics")
        if topics is None and kwargs.get("query"):
            topics = [str(kwargs["query"])]

        state = invoke_daily_brief(
            topics=topics,
            days=int(kwargs.get("days", 7)),
            brief_date=kwargs.get("date"),
            max_papers_per_topic=int(kwargs.get("max_papers_per_topic", 4)),
            max_repos_per_topic=int(kwargs.get("max_repos_per_topic", 4)),
            selected_papers=int(kwargs.get("selected_papers", 4)),
            selected_repos=int(kwargs.get("selected_repos", 3)),
        )

        return [
            {
                "type": "daily_brief",
                "source": "daily_brief_agent",
                "date": state["date"],
                "days": state["days"],
                "topics": state["topics"],
                "paper_count": len(state["paper_results"]),
                "repo_count": len(state["repo_results"]),
                "output_path": state["output_path"],
            }
        ]
