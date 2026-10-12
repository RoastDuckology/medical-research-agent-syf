"""Tests for the LangGraph workflow."""

import agent.nodes.research_node as research_node_module
import agent.nodes.report_node as report_node_module
from agent.workflow import invoke
from llm.base import BaseLLM
from rag.indexer import build_or_update_index
from rag.vector_store import reset_vector_store


class FakeLLM(BaseLLM):
    """Small deterministic LLM for workflow tests."""

    def generate(self, prompt: str) -> str:
        return (
            "# LLM Research Report\n\n"
            "需要进一步核查。Source: https://arxiv.org/abs/example\n"
            "file: 2026-07-10.md"
        )


def test_workflow_returns_memory_research_results_and_report(tmp_path, monkeypatch):
    class FakeTool:
        def run(self, **kwargs):
            return [{"type": "paper", "source": "arxiv", "url": "https://arxiv.org/abs/example"}]

    class FakeRegistry:
        def get(self, name):
            return FakeTool()

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

    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("EXTRA_MARKDOWN_FILES", "")
    monkeypatch.setenv("PAPER_NOTES_DIR", str(notes_dir))
    monkeypatch.setenv("PAPERS_DIR", str(papers_dir))
    monkeypatch.setenv("PAPER_TEXT_DIR", str(paper_text_dir))
    monkeypatch.setenv("ENABLE_PDF_INGESTION", "0")
    monkeypatch.setenv("VECTOR_DB_PATH", str(vector_dir))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_workflow_logs")
    monkeypatch.setattr(
        research_node_module,
        "create_default_tool_registry",
        lambda: FakeRegistry(),
    )
    monkeypatch.setattr(report_node_module, "get_llm", lambda: FakeLLM())
    reset_vector_store()
    build_or_update_index()

    final_state = invoke(
        {
            "query": "search arxiv StereoVLA papers",
            "memory": "",
            "retrieved_documents": [],
            "research_results": [],
            "report": "",
        }
    )

    assert final_state["retrieved_documents"]
    assert final_state["research_results"]
    assert "LLM Research Report" in final_state["report"]
    assert "StereoVLA" in final_state["memory"]
