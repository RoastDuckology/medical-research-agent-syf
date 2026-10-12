"""Shared scoring utilities for heterogeneous agent memory."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable


DEFAULT_HALF_LIFE_DAYS = 90.0


def tokenize_for_memory(text: str) -> list[str]:
    """Tokenize mixed Chinese/English research text for lexical memory scoring."""

    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9_\-+.]*", lowered)
    cjk_spans = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    for span in cjk_spans:
        if len(span) <= 8:
            tokens.append(span)
        tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
    return tokens


def lexical_overlap_score(query: str, text: str) -> float:
    """Return simple query-token coverage in text, normalized to 0..1."""

    query_tokens = set(tokenize_for_memory(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize_for_memory(text))
    hits = sum(1 for token in query_tokens if token in text_tokens)
    return hits / len(query_tokens)


def bm25_scores(query: str, documents: Iterable[str]) -> list[float]:
    """Compute lightweight BM25 scores for a small in-memory candidate set."""

    corpus_tokens = [tokenize_for_memory(document) for document in documents]
    query_tokens = tokenize_for_memory(query)
    if not query_tokens or not corpus_tokens:
        return [0.0 for _ in corpus_tokens]

    doc_freq: Counter[str] = Counter()
    for tokens in corpus_tokens:
        doc_freq.update(set(tokens))

    document_count = len(corpus_tokens)
    average_length = sum(len(tokens) for tokens in corpus_tokens) / max(document_count, 1)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens in corpus_tokens:
        term_freq = Counter(tokens)
        doc_length = max(len(tokens), 1)
        score = 0.0
        for token in query_tokens:
            if token not in term_freq:
                continue
            idf = math.log(1 + (document_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            numerator = term_freq[token] * (k1 + 1)
            denominator = term_freq[token] + k1 * (1 - b + b * doc_length / max(average_length, 1))
            score += idf * numerator / denominator
        scores.append(score)
    return scores


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize non-negative scores to 0..1."""

    if not scores:
        return []
    high = max(scores)
    low = min(scores)
    if high <= 0:
        return [0.0 for _ in scores]
    if math.isclose(high, low):
        return [1.0 if score > 0 else 0.0 for score in scores]
    return [(score - low) / (high - low) for score in scores]


def distance_to_similarity(distance: float | int | None) -> float:
    """Convert Chroma distance to a bounded similarity value."""

    if distance is None:
        return 0.0
    numeric = max(float(distance), 0.0)
    return 1.0 / (1.0 + numeric)


def time_decay(timestamp: str, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Return exponential recency decay. Missing/unparseable dates do not penalize."""

    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return 1.0
    now = datetime.now(timezone.utc)
    age_days = max((now - parsed).total_seconds() / 86400.0, 0.0)
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def importance_factor(importance: float) -> float:
    """Map 0..1 importance to the chapter-8 style multiplier."""

    bounded = max(0.0, min(1.0, float(importance)))
    return 0.8 + bounded * 0.4


def final_memory_score(
    base_similarity: float,
    timestamp: str = "",
    importance: float = 0.5,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Score memory on a shared 0..~1 scale after type-specific base scoring."""

    base = max(0.0, min(1.0, float(base_similarity)))
    return base * time_decay(timestamp, half_life_days) * importance_factor(importance)


def _parse_timestamp(value: str) -> datetime | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    candidates = [cleaned]
    if len(cleaned) >= 10:
        candidates.append(cleaned[:10])
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None
