"""Retriever for querying the persistent local Chroma index."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings
from rag.vector_store import create_or_load_vector_store, vector_store_count


logger = logging.getLogger(__name__)


def retrieve_documents(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant document chunks from the local Chroma database."""

    normalized_query = query.strip()
    if not normalized_query:
        logger.warning("Empty query received by retriever.")
        return []

    if vector_store_count() == 0:
        logger.warning("Chroma vector store is empty.")
        return []

    settings = get_settings()
    k = top_k if top_k is not None else settings.retrieval_top_k
    vector_store = create_or_load_vector_store()

    # Chroma returns a distance score here. Lower distance means closer match;
    # it is not a direct similarity probability.
    results = vector_store.similarity_search_with_score(normalized_query, k=k)
    if not results:
        logger.warning("No retrieval results found for query: %s", normalized_query)
        return []

    retrieved: list[dict[str, Any]] = []
    for document, score in results:
        retrieved.append(
            {
                "content": document.page_content,
                "source": document.metadata.get("source", ""),
                "filename": document.metadata.get("filename", ""),
                "date": document.metadata.get("date", ""),
                "score": float(score),
            }
        )

    return retrieved
