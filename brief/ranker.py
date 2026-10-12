"""Filtering and ranking helpers for concise daily briefs."""

from __future__ import annotations

from datetime import datetime, timezone

from brief.state import BriefItem
from brief.topics import TOP_VENUE_HINTS


def select_brief_items(
    papers: list[BriefItem],
    repositories: list[BriefItem],
    *,
    max_papers: int = 6,
    max_repositories: int = 5,
) -> tuple[list[BriefItem], list[BriefItem]]:
    """Filter source-grounded items and keep the most useful brief candidates."""

    selected_papers = sorted(
        [item for item in papers if item.get("url") and item.get("title")],
        key=_paper_score,
        reverse=True,
    )[:max_papers]
    selected_repositories = sorted(
        [item for item in repositories if item.get("url") and item.get("title")],
        key=_repository_score,
        reverse=True,
    )[:max_repositories]
    return selected_papers, selected_repositories


def _paper_score(item: BriefItem) -> float:
    text = " ".join(
        [
            item.get("title", ""),
            item.get("summary", ""),
            item.get("venue_hint", ""),
        ]
    ).lower()
    score = 0.0
    score += 2.0 if item.get("published") else 0.0
    score += min(len(item.get("summary", "")) / 500.0, 2.0)
    score += 3.0 if any(venue.lower() in text for venue in TOP_VENUE_HINTS) else 0.0
    score += _topic_match_bonus(item)
    return score


def _repository_score(item: BriefItem) -> float:
    score = 0.0
    score += min(float(item.get("stars", 0)) / 1000.0, 5.0)
    score += 1.0 if item.get("language") else 0.0
    score += 1.5 if item.get("summary") else 0.0
    score += _recency_bonus(item.get("updated_at", ""))
    score += _topic_match_bonus(item)
    return score


def _topic_match_bonus(item: BriefItem) -> float:
    topic = item.get("topic", "").lower()
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    if topic and topic in text:
        return 2.0
    return 0.0

def _recency_bonus(updated_at: str) -> float:
    if not updated_at:
        return 0.0
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_days = (datetime.now(timezone.utc) - updated).days
    if age_days <= 1:
        return 2.0
    if age_days <= 7:
        return 1.5
    if age_days <= 15:
        return 1.0
    return 0.0
