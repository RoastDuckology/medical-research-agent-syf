"""Metrics for source-grounded RAG and agent evaluation."""

from __future__ import annotations

import re
from typing import Any


URL_RE = re.compile(r"https?://\S+")


def evaluate_retrieval_case(
    case: dict[str, Any],
    retrieved_documents: list[dict[str, Any]],
    k_values: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    """Evaluate one RAG case against retrieved document chunks."""

    expected_sources = [str(item).lower() for item in case.get("expected_sources", [])]
    expected_keywords = [str(item).lower() for item in case.get("expected_keywords", [])]
    expected_pages = [int(item) for item in case.get("expected_pages", [])]
    relevant_ranks = [
        index
        for index, document in enumerate(retrieved_documents, start=1)
        if _is_relevant(document, expected_sources, expected_keywords, expected_pages)
    ]
    no_relevant_expected = bool(case.get("no_relevant_expected")) or (
        not expected_sources and not expected_keywords and not expected_pages
    )
    page_ranks = [
        index
        for index, document in enumerate(retrieved_documents, start=1)
        if _page_matches(document, expected_pages)
    ]

    metrics: dict[str, Any] = {
        "question": case.get("question", ""),
        "category": case.get("category", ""),
        "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
        "mrr": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "retrieved_count": len(retrieved_documents),
        "false_positive": (1.0 if retrieved_documents else 0.0) if no_relevant_expected else None,
        "pdf_page_hit": (1.0 if page_ranks else 0.0) if expected_pages else None,
    }
    for k in k_values:
        metrics[f"hit@{k}"] = 1.0 if any(rank <= k for rank in relevant_ranks) else 0.0
        metrics[f"recall@{k}"] = metrics[f"hit@{k}"]
    return metrics


def citation_coverage(answer: str, required_points: list[str] | None = None) -> float:
    """Estimate whether important answer points have visible source links."""

    points = [point for point in (required_points or []) if point.strip()]
    if not points:
        return 1.0 if URL_RE.search(answer) else 0.0

    covered = 0
    lines = answer.splitlines()
    for point in points:
        for line in lines:
            if point in line and URL_RE.search(line):
                covered += 1
                break
    return covered / len(points)


def aggregate_metrics(case_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Average numeric metrics over cases."""

    if not case_metrics:
        return {}

    numeric_keys = sorted(
        {
            key
            for item in case_metrics
            for key, value in item.items()
            if isinstance(value, int | float)
        }
        - {"question"}
    )
    aggregate: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(item[key]) for item in case_metrics if isinstance(item.get(key), int | float)]
        if values:
            aggregate[key] = sum(values) / len(values)
    return aggregate


def _is_relevant(
    document: dict[str, Any],
    expected_sources: list[str],
    expected_keywords: list[str],
    expected_pages: list[int],
) -> bool:
    source_blob = " ".join(
        str(document.get(key, ""))
        for key in ("source", "filename", "document_type", "heading_path")
    ).lower()
    content = str(document.get("content", "")).lower()

    source_ok = not expected_sources or any(source in source_blob for source in expected_sources)
    keyword_ok = not expected_keywords or any(keyword in content or keyword in source_blob for keyword in expected_keywords)
    page_ok = True
    if expected_pages:
        page_ok = _page_matches(document, expected_pages)
    return source_ok and keyword_ok and page_ok


def _page_matches(document: dict[str, Any], expected_pages: list[int]) -> bool:
    if not expected_pages:
        return False
    page_start = _safe_int(document.get("page_start"))
    page_end = _safe_int(document.get("page_end")) or page_start
    return any(page_start <= page <= page_end for page in expected_pages)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
