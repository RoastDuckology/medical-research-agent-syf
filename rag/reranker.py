"""Lightweight reranking utilities for local RAG candidates."""

from __future__ import annotations

import math
from typing import Any

from memory.scoring import tokenize_for_memory


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Rerank candidates with a cross-encoder-style lexical interaction score.

    This intentionally stays dependency-light. It approximates the second-stage
    reranker role by scoring query-document token overlap, phrase hits, metadata
    matches, and the first-stage hybrid score.
    """

    if not candidates:
        return []

    query_tokens = tokenize_for_memory(query)
    query_set = set(query_tokens)
    if not query_set:
        return candidates[:top_k]

    rescored: list[dict[str, Any]] = []
    for item in candidates:
        document = item.get("document")
        content = str(getattr(document, "page_content", ""))
        metadata = getattr(document, "metadata", {}) or {}
        lexical = _lexical_interaction_score(query, query_set, content, metadata)
        first_stage = float(item.get("memory_score", item.get("score", 0.0)))
        rerank_score = (0.55 * lexical) + (0.45 * _squash(first_stage))
        copied = dict(item)
        copied["rerank_score"] = rerank_score
        copied["memory_score"] = rerank_score
        copied["score"] = rerank_score
        rescored.append(copied)

    rescored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return rescored[:top_k]


def _lexical_interaction_score(
    query: str,
    query_set: set[str],
    content: str,
    metadata: dict[str, Any],
) -> float:
    content_lower = content.lower()
    metadata_blob = " ".join(str(value) for value in metadata.values()).lower()
    doc_tokens = set(tokenize_for_memory(content_lower + " " + metadata_blob))
    if not doc_tokens:
        return 0.0

    overlap = len(query_set & doc_tokens) / max(1, len(query_set))
    phrase_bonus = 0.0
    normalized_query = query.lower().strip()
    if normalized_query and normalized_query in content_lower:
        phrase_bonus += 0.25
    if normalized_query and normalized_query in metadata_blob:
        phrase_bonus += 0.15

    metadata_bonus = 0.0
    important_fields = (
        "filename",
        "paper_title",
        "heading_path",
        "section_title",
        "document_type",
    )
    for field in important_fields:
        value_tokens = set(tokenize_for_memory(str(metadata.get(field, "")).lower()))
        if value_tokens & query_set:
            metadata_bonus += 0.04

    return min(1.0, overlap + phrase_bonus + metadata_bonus)


def _squash(score: float) -> float:
    if score <= 0:
        return 0.0
    return score / (1.0 + math.fabs(score))
