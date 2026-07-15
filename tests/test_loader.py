"""Tests for Markdown document loading."""

from rag.loader import load_markdown_documents


def test_load_markdown_documents_reads_metadata(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "2026-07-10.md").write_text("# 2026-07-10\n\nStereoVLA 测试", encoding="utf-8")

    documents = load_markdown_documents(str(logs_dir))

    assert len(documents) == 1
    assert "StereoVLA" in documents[0].page_content
    assert documents[0].metadata["filename"] == "2026-07-10.md"
    assert documents[0].metadata["date"] == "2026-07-10"
    assert documents[0].metadata["document_type"] == "daily_log"


def test_load_markdown_documents_empty_directory(tmp_path):
    assert load_markdown_documents(str(tmp_path)) == []
