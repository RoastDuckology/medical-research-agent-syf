"""Tests for local Chroma retrieval."""

from rag.indexer import build_or_update_index
from rag.retriever import retrieve_documents
from rag.vector_store import reset_vector_store


def test_retrieve_stereovla_and_lewm_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    vector_dir = tmp_path / "vector_db"
    notes_dir = tmp_path / "paper_notes"
    papers_dir = tmp_path / "papers"
    paper_text_dir = tmp_path / "paper_text"
    logs_dir.mkdir()
    notes_dir.mkdir()
    papers_dir.mkdir()
    (logs_dir / "2026-07-10.md").write_text(
        "# 2026-07-10\n\n今天完成 StereoVLA 特征提取接口测试。",
        encoding="utf-8",
    )
    (logs_dir / "2026-07-12.md").write_text(
        "# 2026-07-12\n\n今天阅读 LeWM，并整理 latent world model。",
        encoding="utf-8",
    )

    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("EXTRA_MARKDOWN_FILES", "")
    monkeypatch.setenv("PAPER_NOTES_DIR", str(notes_dir))
    monkeypatch.setenv("PAPERS_DIR", str(papers_dir))
    monkeypatch.setenv("PAPER_TEXT_DIR", str(paper_text_dir))
    monkeypatch.setenv("ENABLE_PDF_INGESTION", "0")
    monkeypatch.setenv("VECTOR_DB_PATH", str(vector_dir))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_retriever_logs")
    monkeypatch.setenv("EMBEDDING_MODEL", "hash")
    reset_vector_store()
    build_or_update_index()

    stereovla_results = retrieve_documents("StereoVLA", top_k=1)
    lewm_results = retrieve_documents("LeWM", top_k=1)

    assert stereovla_results
    assert "StereoVLA" in stereovla_results[0]["content"]
    assert stereovla_results[0]["memory_score"] > 0
    assert "dense_score" in stereovla_results[0]
    assert "keyword_score" in stereovla_results[0]
    assert lewm_results
    assert "LeWM" in lewm_results[0]["content"]


def test_retrieve_empty_query_returns_empty_list():
    assert retrieve_documents("", top_k=1) == []


def test_retrieve_keyword_match_can_surface_document(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    vector_dir = tmp_path / "vector_db"
    notes_dir = tmp_path / "paper_notes"
    papers_dir = tmp_path / "papers"
    paper_text_dir = tmp_path / "paper_text"
    logs_dir.mkdir()
    notes_dir.mkdir()
    papers_dir.mkdir()
    (logs_dir / "2026-07-20.md").write_text(
        "# 2026-07-20\n\nAblation keyword_token_rare_xyz appears only here.",
        encoding="utf-8",
    )
    (logs_dir / "2026-07-21.md").write_text(
        "# 2026-07-21\n\nGeneral surgical world model note.",
        encoding="utf-8",
    )

    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("EXTRA_MARKDOWN_FILES", "")
    monkeypatch.setenv("PAPER_NOTES_DIR", str(notes_dir))
    monkeypatch.setenv("PAPERS_DIR", str(papers_dir))
    monkeypatch.setenv("PAPER_TEXT_DIR", str(paper_text_dir))
    monkeypatch.setenv("ENABLE_PDF_INGESTION", "0")
    monkeypatch.setenv("VECTOR_DB_PATH", str(vector_dir))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_retriever_keyword")
    monkeypatch.setenv("EMBEDDING_MODEL", "hash")
    reset_vector_store()
    build_or_update_index()

    results = retrieve_documents(
        "keyword_token_rare_xyz",
        top_k=1,
        expand=False,
        domain_guard=False,
    )

    assert results
    assert "keyword_token_rare_xyz" in results[0]["content"]
    assert results[0]["keyword_score"] > 0
