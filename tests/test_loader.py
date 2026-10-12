"""Tests for Markdown document loading."""

from rag.loader import load_markdown_documents, load_research_documents


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


def test_load_research_documents_reads_extra_markdown_file(tmp_path):
    logs_dir = tmp_path / "logs"
    notes_dir = tmp_path / "paper_notes"
    logs_dir.mkdir()
    extra_file = tmp_path / "实验进度记录.md"
    extra_file.write_text("## 2026-07-24\n\n### 当前判断\n\nStereo data audit passed.", encoding="utf-8")

    documents = load_research_documents(str(logs_dir), [extra_file], notes_dir)

    assert len(documents) == 1
    assert documents[0].metadata["filename"] == "实验进度记录.md"
    assert documents[0].metadata["document_type"] == "experiment_progress_log"
    assert "Stereo data audit passed" in documents[0].page_content


def test_load_research_documents_reads_paper_notes(tmp_path):
    logs_dir = tmp_path / "logs"
    notes_dir = tmp_path / "paper_notes"
    logs_dir.mkdir()
    notes_dir.mkdir()
    (notes_dir / "paper_corpus_index.md").write_text(
        "# Paper Corpus\n\n## LeWorldModel\n\nSource: https://arxiv.org/abs/2603.19312",
        encoding="utf-8",
    )

    documents = load_research_documents(str(logs_dir), [], notes_dir)

    assert len(documents) == 1
    assert documents[0].metadata["filename"] == "paper_corpus_index.md"
    assert documents[0].metadata["document_type"] == "paper_note"
    assert "LeWorldModel" in documents[0].page_content


def test_load_research_documents_can_disable_pdf_ingestion(tmp_path):
    logs_dir = tmp_path / "logs"
    notes_dir = tmp_path / "paper_notes"
    papers_dir = tmp_path / "papers"
    text_dir = tmp_path / "paper_text"
    logs_dir.mkdir()
    notes_dir.mkdir()
    papers_dir.mkdir()

    documents = load_research_documents(
        str(logs_dir),
        [],
        notes_dir,
        papers_dir,
        text_dir,
        enable_pdf_ingestion=False,
    )

    assert documents == []
