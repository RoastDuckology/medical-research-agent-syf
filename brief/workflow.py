"""Standalone workflow for the daily AI brief agent."""

from __future__ import annotations

import argparse
from datetime import date

from brief.collectors import collect_recent_arxiv_papers, collect_recent_github_repositories
from brief.formatter import format_daily_brief
from brief.ranker import select_brief_items
from brief.state import DailyBriefState
from brief.storage import save_daily_brief
from brief.summarizer import summarize_selected_items
from brief.topics import DEFAULT_TOPICS


def invoke_daily_brief(
    *,
    topics: list[str] | None = None,
    days: int = 7,
    brief_date: str | None = None,
    max_papers_per_topic: int = 4,
    max_repos_per_topic: int = 4,
    selected_papers: int = 4,
    selected_repos: int = 3,
) -> DailyBriefState:
    """Generate and save a daily AI research brief."""

    selected_topics = topics or DEFAULT_TOPICS
    selected_date = brief_date or date.today().isoformat()

    paper_results = collect_recent_arxiv_papers(
        topics=selected_topics,
        days=days,
        max_results_per_topic=max_papers_per_topic,
    )
    repo_results = collect_recent_github_repositories(
        topics=selected_topics,
        days=days,
        max_results_per_topic=max_repos_per_topic,
    )
    brief_papers, brief_repositories = select_brief_items(
        paper_results,
        repo_results,
        max_papers=selected_papers,
        max_repositories=selected_repos,
    )
    summarized_papers = summarize_selected_items(brief_papers, item_type="paper")
    summarized_repositories = summarize_selected_items(brief_repositories, item_type="repository")
    brief = format_daily_brief(
        brief_date=selected_date,
        days=days,
        topics=selected_topics,
        papers=summarized_papers,
        repositories=summarized_repositories,
    )
    output_path = save_daily_brief(brief, selected_date)

    return {
        "date": selected_date,
        "days": days,
        "topics": selected_topics,
        "paper_results": paper_results,
        "repo_results": repo_results,
        "brief": brief,
        "output_path": output_path,
    }


def main() -> None:
    """CLI entry point for `python -m brief.workflow`."""

    parser = argparse.ArgumentParser(description="Generate a daily AI research brief.")
    parser.add_argument("--days", type=int, default=7, help="Recent time window in days.")
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="Topic to track. Can be passed multiple times.",
    )
    parser.add_argument("--date", default=None, help="Brief date, defaults to today.")
    parser.add_argument("--max-papers", type=int, default=4, help="Max arXiv papers per topic.")
    parser.add_argument("--max-repos", type=int, default=4, help="Max GitHub repos per topic.")
    parser.add_argument("--selected-papers", type=int, default=4, help="Max papers kept in the final brief.")
    parser.add_argument("--selected-repos", type=int, default=3, help="Max repositories kept in the final brief.")
    args = parser.parse_args()

    state = invoke_daily_brief(
        topics=args.topics,
        days=args.days,
        brief_date=args.date,
        max_papers_per_topic=args.max_papers,
        max_repos_per_topic=args.max_repos,
        selected_papers=args.selected_papers,
        selected_repos=args.selected_repos,
    )
    print(f"Daily brief saved to: {state['output_path']}")


if __name__ == "__main__":
    main()
