"""Embedding model factory for local vector retrieval."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings
from memory.scoring import tokenize_for_memory


class HashEmbeddings(Embeddings):
    """Small deterministic embedding backend for tests and low-resource smoke runs."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize_for_memory(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Load the configured local Hugging Face embedding model."""

    settings = get_settings()
    if settings.embedding_model.lower() in {"hash", "local-hash", "test-hash"}:
        return HashEmbeddings()

    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        message = (
            f"Failed to load embedding model '{settings.embedding_model}'. "
            "Check network access for the first model download, or set "
            "EMBEDDING_MODEL to a locally cached sentence-transformers model."
        )
        raise RuntimeError(message) from exc


def clear_embedding_model_cache() -> None:
    """Clear cached embedding model after tests or environment changes."""

    get_embedding_model.cache_clear()
