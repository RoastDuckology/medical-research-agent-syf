"""Tests for PDF text cache loading."""

import os
from pathlib import Path

from rag.pdf_loader import load_pdf_documents


def test_load_pdf_documents_uses_markdown_cache(tmp_path):
    papers_dir = tmp_path / "papers"
    text_dir = tmp_path / "paper_text"
    papers_dir.mkdir()
    text_dir.mkdir()
    pdf_path = papers_dir / "2301.08243_i_jepa.pdf"
    cache_path = text_dir / "2301.08243_i_jepa.md"
    manifest_path = papers_dir / "manifest.jsonl"

    pdf_path.write_bytes(b"%PDF- fake placeholder")
    cache_path.write_text(
        "# I-JEPA\n\n## Page 1\n\nJoint-Embedding Predictive Architecture text.",
        encoding="utf-8",
    )
    os.utime(cache_path, (pdf_path.stat().st_mtime + 5, pdf_path.stat().st_mtime + 5))
    manifest_path.write_text(
        '{"arxiv_id":"2301.08243","title":"I-JEPA","pdf_file":"data/papers/2301.08243_i_jepa.pdf","abs_url":"https://arxiv.org/abs/2301.08243","pdf_url":"https://arxiv.org/pdf/2301.08243"}\n',
        encoding="utf-8",
    )

    documents = load_pdf_documents(papers_dir, text_dir, manifest_path)

    assert len(documents) == 1
    assert documents[0].metadata["document_type"] == "paper_pdf"
    assert documents[0].metadata["page_start"] == 1
    assert documents[0].metadata["arxiv_id"] == "2301.08243"
    assert "Joint-Embedding" in documents[0].page_content
