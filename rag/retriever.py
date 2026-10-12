"""Public retrieval entry point for the persistent local RAG index."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings
from memory.scoring import tokenize_for_memory
from rag.pipeline import RetrievalPipeline
from rag.vector_store import create_or_load_vector_store, vector_store_count


logger = logging.getLogger(__name__)


def retrieve_documents(
    query: str,
    top_k: int | None = None,
    expand: bool = True,
    document_types: list[str] | tuple[str, ...] | None = None,
    enable_dense: bool = True,
    enable_bm25: bool = True,
    min_score: float | None = None,
    domain_guard: bool | None = None,
    candidate_k: int | None = None,
    rerank: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve relevant document chunks from the local RAG database."""

    normalized_query = query.strip()
    if not normalized_query:
        logger.warning("Empty query received by retriever.")
        return []

    if vector_store_count() == 0:
        logger.warning("Chroma vector store is empty.")
        return []

    settings = get_settings()
    guard_enabled = settings.retrieval_domain_guard if domain_guard is None else bool(domain_guard)
    if guard_enabled and not _passes_domain_guard(normalized_query):
        logger.warning("Query is outside the configured research domain: %s", normalized_query)
        return []

    pipeline = RetrievalPipeline(
        vector_store=create_or_load_vector_store(),
        top_k=top_k if top_k is not None else settings.retrieval_top_k,
        document_types=tuple(document_types or ()),
        enable_dense=enable_dense,
        enable_bm25=enable_bm25,
        min_score=settings.retrieval_min_score if min_score is None else float(min_score),
        candidate_k=candidate_k,
        rerank=rerank,
    )
    results = pipeline.retrieve(normalized_query, expand=expand)
    if not results:
        logger.warning("No retrieval results found for query: %s", normalized_query)
    return results


DOMAIN_TERMS = {
    "stereovla",
    "lewm",
    "leworldmodel",
    "foundationstereo",
    "endo-ttap",
    "endotap",
    "endo",
    "jepa",
    "i-jepa",
    "v-jepa",
    "ff-jepa",
    "surgvla",
    "surgvla-bench",
    "cem",
    "vla",
    "rag",
    "bm25",
    "chroma",
    "context",
    "citation",
    "fallback",
    "route",
    "routes",
    "current",
    "phase",
    "traceability",
    "conclusion",
    "blocker",
    "next",
    "step",
    "world",
    "planning",
    "disparity",
    "stereo",
    "tracking",
    "laparoscopic",
    "surgical",
    "surgery",
    "medical",
    "vision",
    "language",
    "action",
    "experiment",
    "baseline",
    "synthetic",
    "bipegtransfer",
    "daily",
    "brief",
    "ai",
    "论文",
    "文献",
    "实验",
    "进度",
    "记录",
    "简报",
    "日报",
    "医疗",
    "手术",
    "双目",
    "视差",
    "跟踪",
    "上下文",
    "引用",
    "记忆",
    "知识库",
    "规划",
    "风险",
    "基线",
}


def _passes_domain_guard(query: str) -> bool:
    normalized = query.lower()
    tokens = set(tokenize_for_memory(normalized))
    if tokens & DOMAIN_TERMS:
        return True
    return any(term in normalized for term in DOMAIN_TERMS if len(term) >= 3)
