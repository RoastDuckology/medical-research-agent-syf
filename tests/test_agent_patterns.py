"""Tests for LangGraph routing and agent design patterns."""

from agent.nodes.daily_brief_node import daily_brief_node
from agent.nodes.planner_node import planner_node
from agent.nodes.reflection_node import reflection_node
from agent.nodes.report_node import report_node
from agent.nodes.research_node import research_node
from agent.workflow import invoke
from llm.base import BaseLLM


def test_planner_selects_arxiv_for_literature_query():
    update = planner_node({"query": "查找 StereoVLA 相关论文和综述"})

    assert update["task_type"] == "research"
    assert "arxiv" in update["selected_tools"]
    assert update["plan"]


def test_planner_selects_github_for_code_query():
    update = planner_node({"query": "查找 surgical world model 开源代码"})

    assert update["task_type"] == "research"
    assert "github" in update["selected_tools"]


def test_planner_routes_daily_brief_query():
    update = planner_node({"query": "生成今日 AI 简报"})

    assert update["task_type"] == "daily_brief"
    assert update["selected_tools"] == ["daily_brief"]


def test_research_node_runs_selected_tools_with_trace(monkeypatch):
    class FakeTool:
        def __init__(self, name):
            self.name = name

        def run(self, **kwargs):
            return [{"type": self.name, "query": kwargs["query"]}]

    class FakeRegistry:
        def get(self, name):
            if name not in {"arxiv", "github"}:
                raise KeyError(name)
            return FakeTool(name)

    monkeypatch.setattr(
        "agent.nodes.research_node.create_default_tool_registry",
        lambda: FakeRegistry(),
    )

    update = research_node(
        {
            "query": "StereoVLA paper",
            "selected_tools": ["arxiv"],
        }
    )

    assert update["research_results"] == [{"type": "arxiv", "query": "StereoVLA paper"}]
    assert update["reasoning_trace"][0]["action"] == "arxiv"


def test_daily_brief_node_returns_output_path_report(monkeypatch):
    class FakeDailyBriefTool:
        def run(self, **kwargs):
            return [
                {
                    "type": "daily_brief",
                    "date": "2026-07-30",
                    "days": 7,
                    "topics": ["medical ai"],
                    "paper_count": 2,
                    "repo_count": 1,
                    "output_path": "data/reports/daily/2026-07-30.md",
                }
            ]

    monkeypatch.setattr("agent.nodes.daily_brief_node.DailyBriefTool", FakeDailyBriefTool)

    update = daily_brief_node({"query": "生成今日 AI 简报"})

    assert update["research_results"][0]["type"] == "daily_brief"
    assert "data/reports/daily/2026-07-30.md" in update["report"]
    assert update["reasoning_trace"][0]["action"] == "daily_brief"


def test_reflection_accepts_daily_brief_with_output_path():
    update = reflection_node(
        {
            "task_type": "daily_brief",
            "report": "# 每日 AI 简报\n\n输出文件：data/reports/daily/2026-07-30.md",
            "research_results": [
                {
                    "type": "daily_brief",
                    "output_path": "data/reports/daily/2026-07-30.md",
                }
            ],
        }
    )

    assert update["reflection"]["passed"]


def test_reflection_appends_notes_when_sources_are_missing():
    update = reflection_node(
        {
            "task_type": "research",
            "report": "# Report\n\nThis is a confident answer.",
            "research_results": [{"url": "https://arxiv.org/abs/example"}],
            "retrieved_documents": [],
        }
    )

    assert not update["reflection"]["passed"]
    assert "report" not in update
    assert any("source links" in issue for issue in update["reflection"]["issues"])


def test_report_node_falls_back_when_configured_llm_fails(monkeypatch):
    class FailingLLM(BaseLLM):
        def generate(self, prompt: str) -> str:
            raise RuntimeError("network failed")

    monkeypatch.setattr("agent.nodes.report_node.get_llm", lambda: FailingLLM())

    update = report_node(
        {
            "query": "StereoVLA",
            "memory": "",
            "research_results": [],
            "report": "",
        }
    )

    assert "MockLLM" in update["report"]
    assert update["report_error"] == "network failed"
    assert update["context_summary"]["packet_count"] > 0


def test_research_workflow_uses_general_path(monkeypatch, tmp_path):
    class FakeLLM(BaseLLM):
        def generate(self, prompt: str) -> str:
            return (
                "# Report\n\n"
                "需要进一步核查。Source: https://arxiv.org/abs/example\n"
                "file: experiment.md"
            )

    class FakeRegistry:
        def get(self, name):
            class FakeTool:
                def run(self, **kwargs):
                    return [{"type": "paper", "url": "https://arxiv.org/abs/example"}]

            return FakeTool()

    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setattr(
        "agent.nodes.memory_node.retrieve_documents",
        lambda query: [
            {
                "filename": "experiment.md",
                "date": "2026-07-24",
                "heading_path": "2026-07-24 > 当前判断",
                "content": "StereoVLA experiment memory.",
                "score": 0.1,
            }
        ],
    )
    monkeypatch.setattr(
        "agent.nodes.research_node.create_default_tool_registry",
        lambda: FakeRegistry(),
    )
    monkeypatch.setattr("agent.nodes.report_node.get_llm", lambda: FakeLLM())

    final_state = invoke(
        {
            "query": "查找 StereoVLA 相关文章",
            "memory": "",
            "memory_records": [],
            "retrieved_documents": [],
            "research_results": [],
            "report": "",
        }
    )

    assert final_state["task_type"] == "research"
    assert final_state["plan"]
    assert final_state["reasoning_trace"]
    assert final_state["reflection"]["passed"]


def test_daily_brief_workflow_uses_conditional_path(monkeypatch):
    class FakeDailyBriefTool:
        def run(self, **kwargs):
            return [
                {
                    "type": "daily_brief",
                    "date": "2026-07-30",
                    "days": 7,
                    "topics": ["生成今日 AI 简报"],
                    "paper_count": 1,
                    "repo_count": 1,
                    "output_path": "data/reports/daily/2026-07-30.md",
                }
            ]

    monkeypatch.setattr("agent.nodes.daily_brief_node.DailyBriefTool", FakeDailyBriefTool)
    monkeypatch.setattr(
        "agent.nodes.memory_node.retrieve_documents",
        lambda query: (_ for _ in ()).throw(AssertionError("memory path should be skipped")),
    )

    final_state = invoke(
        {
            "query": "生成今日 AI 简报",
            "memory": "",
            "memory_records": [],
            "retrieved_documents": [],
            "research_results": [],
            "report": "",
        }
    )

    assert final_state["task_type"] == "daily_brief"
    assert final_state["reflection"]["passed"]
    assert "data/reports/daily/2026-07-30.md" in final_state["report"]
