"""Chroma vector store helpers for the local RAG index."""

from __future__ import annotations

import hashlib
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import get_settings
from rag.embedding import clear_embedding_model_cache, get_embedding_model


logger = logging.getLogger(__name__)


def create_or_load_vector_store() -> Chroma:
    """Create or load the persistent Chroma vector store."""

    settings = get_settings()
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    return _cached_vector_store(
        settings.chroma_collection_name,
        str(settings.vector_db_path),
    )


@lru_cache(maxsize=4)
def _cached_vector_store(collection_name: str, persist_directory: str) -> Chroma:
    """Return a cached Chroma client for Streamlit reruns and repeated queries."""

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=persist_directory,
    )


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document_id(document: Document) -> str:
    source = str(document.metadata.get("source", ""))
    chunk_index = str(document.metadata.get("chunk_index", ""))
    page_start = str(document.metadata.get("page_start", ""))
    page_end = str(document.metadata.get("page_end", ""))
    content_digest = _content_hash(document.page_content)
    raw_id = f"{source}|{page_start}|{page_end}|{chunk_index}|{content_digest}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def index_documents(documents: list[Document]) -> dict[str, Any]:
    """Index document chunks into Chroma with simple duplicate handling."""

    vector_store = create_or_load_vector_store()
    added = 0
    skipped = 0
    updated = 0

    for document in documents:
        content_digest = _content_hash(document.page_content)
        document.metadata["content_hash"] = content_digest
        document_id = _document_id(document)
        source = str(document.metadata.get("source", ""))
        chunk_index = document.metadata.get("chunk_index")
        page_start = document.metadata.get("page_start", "")
        page_end = document.metadata.get("page_end", "")

        existing = vector_store.get(where={"source": source}, include=["metadatas"])
        existing_ids = existing.get("ids", [])
        existing_metadatas = existing.get("metadatas", [])

        duplicate_found = False
        stale_ids: list[str] = []
        for existing_id, metadata in zip(existing_ids, existing_metadatas):
            if metadata.get("chunk_index") != chunk_index:
                continue
            if metadata.get("page_start", "") != page_start:
                continue
            if metadata.get("page_end", "") != page_end:
                continue
            if metadata.get("content_hash") == content_digest:
                duplicate_found = True
                break
            stale_ids.append(existing_id)

        if duplicate_found:
            skipped += 1
            continue

        if stale_ids:
            vector_store.delete(ids=stale_ids)
            updated += len(stale_ids)

        vector_store.add_documents([document], ids=[document_id])
        added += 1

    return {
        "added": added,
        "skipped": skipped,
        "updated": updated,
        "persist_directory": str(get_settings().vector_db_path),
    }


def reset_vector_store() -> None:
    """Delete the local Chroma persistence directory."""

    _cached_vector_store.cache_clear()
    clear_embedding_model_cache()
    vector_db_path = Path(get_settings().vector_db_path)
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)
        logger.info("Removed vector store directory: %s", vector_db_path)


def vector_store_count() -> int:
    """Return the number of stored Chroma records."""

    vector_store = create_or_load_vector_store()
    return int(vector_store._collection.count())
