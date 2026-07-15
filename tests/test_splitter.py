"""Tests for document splitting."""

from langchain_core.documents import Document

from rag.splitter import split_documents


def test_split_documents_preserves_metadata_and_adds_chunk_index(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "30")
    monkeypatch.setenv("CHUNK_OVERLAP", "5")
    document = Document(
        page_content="StereoVLA " * 20,
        metadata={"source": "test.md", "filename": "test.md", "date": "2026-07-10"},
    )

    chunks = split_documents([document])

    assert chunks
    assert chunks[0].metadata["filename"] == "test.md"
    assert chunks[0].metadata["chunk_index"] == 0
