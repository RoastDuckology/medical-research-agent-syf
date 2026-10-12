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


def test_split_documents_preserves_markdown_heading_metadata(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "80")
    document = Document(
        page_content=(
            "## 2026-07-24（dataset audit）\n\n"
            "### 当前判断\n\n"
            "Synthetic stereo shift10 audit passed.\n\n"
            "### 未解决问题\n\n"
            "True 3D stereo rendering is still unresolved."
        ),
        metadata={
            "source": "实验进度记录.md",
            "filename": "实验进度记录.md",
            "date": "",
            "document_type": "experiment_progress_log",
        },
    )

    chunks = split_documents([document])

    assert any(chunk.metadata["date"] == "2026-07-24" for chunk in chunks)
    assert any("当前判断" in chunk.metadata["heading_path"] for chunk in chunks)
    assert any("未解决问题" in chunk.metadata["section_title"] for chunk in chunks)
    assert all("chunk_preview" in chunk.metadata for chunk in chunks)
