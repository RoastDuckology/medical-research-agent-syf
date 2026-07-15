"""Text splitting utilities for local Markdown documents."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks while preserving metadata."""

    if not documents:
        return []

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks: list[Document] = []
    for document in documents:
        document_chunks = splitter.split_documents([document])
        for chunk_index, chunk in enumerate(document_chunks):
            chunk.metadata = {
                **document.metadata,
                **chunk.metadata,
                "chunk_index": chunk_index,
            }
            chunks.append(chunk)

    return chunks
