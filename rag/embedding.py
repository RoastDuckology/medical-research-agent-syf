"""Embedding model factory for local vector retrieval."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the configured local Hugging Face embedding model."""

    settings = get_settings()
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
