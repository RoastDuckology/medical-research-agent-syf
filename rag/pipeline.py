"""Extensible retrieval pipeline for local research RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from memory.scoring import (
    bm25_scores,
    distance_to_similarity,
    final_memory_score,
    normalize_scores,
)
from rag.query_expander import expand_query
from rag.reranker import rerank_candidates


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalPipeline:
    """Hybrid retrieval pipeline with MQE, dense search, and BM25 search."""

    vector_store: Any
    top_k: int
    document_types: tuple[str, ...] = ()
    dense_weight: float = 0.7
    keyword_weight: float = 0.3
    enable_dense: bool = True
    enable_bm25: bool = True
    min_score: float = 0.0
    candidate_k: int | None = None
    rerank: bool = False

    def retrieve(self, query: str, expand: bool = True) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        search_filter = _document_type_filter(self.document_types)
        candidates = {}
        if self.enable_dense:
            candidates.update(self._dense_candidates(normalized_query, expand, search_filter))
        if self.enable_bm25:
            for document, keyword_score in self._keyword_candidates(normalized_query, search_filter):
                key = _document_key(document)
                candidate = candidates.setdefault(
                    key,
                    {"document": document, "dense_similarity": 0.0, "keyword_score": 0.0},
                )
                candidate["keyword_score"] = max(float(candidate["keyword_score"]), keyword_score)

        if not candidates:
            return []

        scored = [self._score_candidate(candidate) for candidate in candidates.values()]
        if self.min_score > 0:
            scored = [item for item in scored if float(item["memory_score"]) >= self.min_score]
        ranked = sorted(scored, key=lambda item: item["memory_score"], reverse=True)
        if self.rerank:
            ranked = rerank_candidates(normalized_query, ranked[: self._candidate_k()], self.top_k)
        return [_format_retrieved(item) for item in ranked[: self.top_k]]

    def _dense_candidates(
        self,
        query: str,
        expand: bool,
        search_filter: dict[str, Any] | None,
    ) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
        candidates: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        query_variants = expand_query(query) if expand else [query]
        for query_variant in query_variants:
            for document, distance in self.vector_store.similarity_search_with_score(
                query_variant,
                k=self._candidate_k(),
                filter=search_filter,
            ):
                key = _document_key(document)
                candidate = candidates.setdefault(
                    key,
                    {"document": document, "dense_similarity": 0.0, "keyword_score": 0.0},
                )
                candidate["dense_similarity"] = max(
                    float(candidate["dense_similarity"]),
                    distance_to_similarity(distance),
                )
                candidate["dense_distance"] = min(
                    float(candidate.get("dense_distance", distance)),
                    float(distance),
                )
        return candidates

    def _keyword_candidates(
        self,
        query: str,
        search_filter: dict[str, Any] | None,
    ) -> list[tuple[Document, float]]:
        raw = self.vector_store.get(
            where=search_filter,
            include=["documents", "metadatas"],
        )
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        if not documents:
            return []

        scores = bm25_scores(query, documents)
        normalized_scores = normalize_scores(scores)
        ranked: list[tuple[Document, float]] = []
        for content, metadata, score in zip(documents, metadatas, normalized_scores):
            if score <= 0:
                continue
            ranked.append((Document(page_content=content, metadata=metadata or {}), float(score)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: max(50, self._candidate_k())]

    def _score_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        document: Document = candidate["document"]
        dense_similarity = float(candidate.get("dense_similarity", 0.0))
        keyword_score = float(candidate.get("keyword_score", 0.0))
        if dense_similarity and keyword_score:
            base_similarity = dense_similarity * self.dense_weight + keyword_score * self.keyword_weight
        else:
            base_similarity = dense_similarity or keyword_score

        importance = _semantic_importance(document.metadata)
        memory_score = final_memory_score(
            base_similarity=base_similarity,
            timestamp=str(document.metadata.get("date", "")),
            importance=importance,
            half_life_days=_semantic_half_life_days(document.metadata),
        )
        return {
            "document": document,
            "score": memory_score,
            "memory_score": memory_score,
            "dense_score": dense_similarity,
            "keyword_score": keyword_score,
            "importance": importance,
            "dense_distance": candidate.get("dense_distance", ""),
        }

    def _candidate_k(self) -> int:
        return max(self.top_k, self.candidate_k or max(self.top_k * 4, 20))


def _format_retrieved(item: dict[str, Any]) -> dict[str, Any]:
    document: Document = item["document"]
    metadata = document.metadata
    return {
        "content": document.page_content,
        "source": metadata.get("source", ""),
        "filename": metadata.get("filename", ""),
        "date": metadata.get("date", ""),
        "document_type": metadata.get("document_type", ""),
        "section_title": metadata.get("section_title", ""),
        "heading_path": metadata.get("heading_path", ""),
        "paper_title": metadata.get("paper_title", ""),
        "arxiv_id": metadata.get("arxiv_id", ""),
        "abs_url": metadata.get("abs_url", ""),
        "pdf_url": metadata.get("pdf_url", ""),
        "page_start": metadata.get("page_start", ""),
        "page_end": metadata.get("page_end", ""),
        "chunk_index": metadata.get("chunk_index", ""),
        "chunk_preview": metadata.get("chunk_preview", ""),
        "score": float(item["score"]),
        "memory_score": float(item["memory_score"]),
        "dense_score": float(item["dense_score"]),
        "keyword_score": float(item["keyword_score"]),
        "importance": float(item["importance"]),
        "dense_distance": item.get("dense_distance", ""),
    }


def _document_key(document: Document) -> tuple[str, str, str, str, str]:
    return (
        str(document.metadata.get("source", "")),
        str(document.metadata.get("page_start", "")),
        str(document.metadata.get("page_end", "")),
        str(document.metadata.get("chunk_index", "")),
        document.page_content[:120],
    )


def _semantic_importance(metadata: dict[str, Any]) -> float:
    document_type = str(metadata.get("document_type", ""))
    if document_type == "experiment_progress_log":
        return 0.9
    if document_type == "paper_note":
        return 0.8
    if document_type == "paper_pdf":
        return 0.7
    return 0.6


def _semantic_half_life_days(metadata: dict[str, Any]) -> float:
    document_type = str(metadata.get("document_type", ""))
    if document_type == "experiment_progress_log":
        return 120.0
    if document_type == "paper_pdf":
        return 365.0
    return 180.0


def _document_type_filter(
    document_types: tuple[str, ...],
) -> dict[str, Any] | None:
    normalized = [item for item in document_types if item]
    if not normalized:
        return None
    if len(normalized) == 1:
        return {"document_type": normalized[0]}
    return {"document_type": {"$in": normalized}}
