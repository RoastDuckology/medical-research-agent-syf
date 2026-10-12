"""State objects for the daily AI brief agent."""

from typing import TypedDict


class BriefItem(TypedDict, total=False):
    """A normalized brief item from an external source."""

    type: str
    source: str
    topic: str
    title: str
    summary: str
    summary_cn: str
    url: str
    authors: list[str]
    published: str
    updated_at: str
    venue_hint: str
    stars: int
    language: str


class DailyBriefState(TypedDict):
    """State passed through the standalone daily brief workflow."""

    date: str
    days: int
    topics: list[str]
    paper_results: list[BriefItem]
    repo_results: list[BriefItem]
    brief: str
    output_path: str
