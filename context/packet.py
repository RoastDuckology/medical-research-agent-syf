"""Typed context packet used before prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContextPacket:
    """One candidate context block with scoring and provenance metadata."""

    source_type: str
    title: str
    content: str
    relevance_score: float = 0.0
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.content) + estimate_tokens(self.title)

    @property
    def priority(self) -> float:
        bounded_relevance = max(0.0, min(1.0, float(self.relevance_score)))
        bounded_importance = max(0.0, min(1.0, float(self.importance)))
        return bounded_relevance * 0.7 + bounded_importance * 0.3


def estimate_tokens(text: str) -> int:
    """Estimate token count for mixed Chinese/English text without model-specific deps."""

    if not text:
        return 0
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_words = len([part for part in text.replace("\n", " ").split(" ") if part.strip()])
    punctuation = max(len(text) - cjk_chars - sum(len(part) for part in text.split()), 0)
    return max(1, int(cjk_chars * 1.1 + ascii_words * 1.3 + punctuation * 0.2))
