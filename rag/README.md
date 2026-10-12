# RAG Module

This module implements the local research knowledge base used by the agent.

## Current Pipeline

```text
Markdown / PDF files
  -> loader.py / pdf_loader.py
  -> splitter.py
  -> vector_store.py  (Chroma + HuggingFace embeddings)
  -> query_expander.py  (lightweight MQE)
  -> pipeline.py  (dense retrieval + BM25 keyword retrieval)
  -> retriever.py  (public entry point)
```

## Implemented

- Markdown and PDF ingestion with `pdfplumber` / `pypdf`.
- Chroma local vector storage.
- Lightweight multi-query expansion (MQE).
- Dense embedding retrieval.
- In-memory BM25 keyword scoring over Chroma candidates.
- Hybrid scoring with `memory_score`, `dense_score`, and `keyword_score`.
- Document type filters for logs, paper notes, and PDF pages.

## Not Implemented Yet

- MarkItDown ingestion.
- Qdrant vector storage backend.
- HyDE hypothetical document generation.
- Cross-encoder reranking.

The public retrieval API is `rag.retriever.retrieve_documents()`. New retrieval strategies
should be added behind `rag.pipeline.RetrievalPipeline` so agent nodes do not need to change.
