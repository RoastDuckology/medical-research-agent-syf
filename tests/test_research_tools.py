"""Tests for standardized research tools."""

import pytest

import agent.nodes.research_node as research_node_module
from tools.arxiv_tool import ArxivTool, _parse_arxiv_response
from tools.github_tool import GithubTool, _parse_github_response
from tools.registry import ToolRegistry, create_default_tool_registry


def test_tool_classes_return_empty_for_empty_query():
    assert ArxivTool().run(query="") == []
    assert GithubTool().run(query="") == []


def test_parse_arxiv_response_returns_expected_shape():
    payload = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2501.00001</id>
        <title> StereoVLA for Surgical Research </title>
        <summary> A paper about StereoVLA. </summary>
        <published>2025-01-01T00:00:00Z</published>
        <author><name>Ada Lovelace</name></author>
      </entry>
    </feed>
    """

    papers = _parse_arxiv_response(payload)

    assert papers
    assert papers[0]["type"] == "paper"
    assert papers[0]["source"] == "arxiv"
    assert papers[0]["title"] == "StereoVLA for Surgical Research"
    assert papers[0]["authors"] == ["Ada Lovelace"]


def test_parse_github_response_returns_expected_shape():
    payload = {
        "items": [
            {
                "full_name": "example/stereovla",
                "description": "StereoVLA toolkit",
                "stargazers_count": 42,
                "html_url": "https://github.com/example/stereovla",
                "language": "Python",
                "updated_at": "2026-07-14T00:00:00Z",
            }
        ]
    }

    repositories = _parse_github_response(payload)

    assert repositories
    assert repositories[0]["type"] == "repository"
    assert repositories[0]["source"] == "github"
    assert repositories[0]["name"] == "example/stereovla"
    assert repositories[0]["stars"] == 42


def test_registry_registers_and_runs_tools():
    registry = create_default_tool_registry()

    assert [tool.name for tool in registry.list_tools()] == ["arxiv", "github"]
    assert registry.get("arxiv").description


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry()
    registry.register(ArxivTool())

    with pytest.raises(ValueError):
        registry.register(ArxivTool())


def test_research_node_updates_research_results(monkeypatch):
    class FakeRegistry:
        def run_all(self, **kwargs):
            return [
                {"type": "paper", "source": "arxiv", "title": kwargs["query"]},
                {"type": "repository", "source": "github", "name": kwargs["query"]},
            ]

    monkeypatch.setattr(
        research_node_module,
        "create_default_tool_registry",
        lambda: FakeRegistry(),
    )

    update = research_node_module.research_node(
        {
            "query": "StereoVLA",
            "memory": "",
            "retrieved_documents": [],
            "research_results": [],
            "report": "",
        }
    )

    assert "research_results" in update
    assert len(update["research_results"]) == 2
