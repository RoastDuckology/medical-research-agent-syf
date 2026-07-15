"""Tests for local Chroma retrieval."""

from rag.indexer import build_or_update_index
from rag.retriever import retrieve_documents
from rag.vector_store import reset_vector_store


def test_retrieve_stereovla_and_lewm_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    vector_dir = tmp_path / "vector_db"
    logs_dir.mkdir()
    (logs_dir / "2026-07-10.md").write_text(
        "# 2026-07-10\n\n今天完成 StereoVLA 特征提取接口测试。",
        encoding="utf-8",
    )
    (logs_dir / "2026-07-12.md").write_text(
        "# 2026-07-12\n\n今天阅读 LeWM，并整理 latent world model。",
        encoding="utf-8",
    )

    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("VECTOR_DB_PATH", str(vector_dir))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_retriever_logs")
    reset_vector_store()
    build_or_update_index()

    stereovla_results = retrieve_documents("StereoVLA", top_k=1)
    lewm_results = retrieve_documents("LeWM", top_k=1)

    assert stereovla_results
    assert "StereoVLA" in stereovla_results[0]["content"]
    assert lewm_results
    assert "LeWM" in lewm_results[0]["content"]


def test_retrieve_empty_query_returns_empty_list():
    assert retrieve_documents("", top_k=1) == []
