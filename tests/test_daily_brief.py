"""Tests for concise DailyBriefAgent behavior."""

from brief.formatter import format_daily_brief
from brief.ranker import select_brief_items
from brief.summarizer import summarize_selected_items
from brief.workflow import invoke_daily_brief


def test_select_brief_items_filters_missing_links_and_limits_results():
    papers = [
        {"title": "No link", "summary": "x"},
        {
            "title": "Relevant paper",
            "url": "https://arxiv.org/abs/1",
            "summary": "medical image segmentation " * 20,
            "topic": "medical image segmentation",
            "published": "2026-07-28T00:00:00Z",
        },
        {
            "title": "Second paper",
            "url": "https://arxiv.org/abs/2",
            "summary": "short",
            "topic": "medical image segmentation",
            "published": "2026-07-27T00:00:00Z",
        },
    ]
    repositories = [
        {"title": "No link repo", "stars": 999},
        {
            "title": "repo/a",
            "url": "https://github.com/repo/a",
            "summary": "medical image segmentation toolkit",
            "topic": "medical image segmentation",
            "stars": 100,
            "updated_at": "2026-07-28T00:00:00Z",
            "language": "Python",
        },
    ]

    selected_papers, selected_repos = select_brief_items(
        papers,
        repositories,
        max_papers=1,
        max_repositories=1,
    )

    assert len(selected_papers) == 1
    assert selected_papers[0]["url"] == "https://arxiv.org/abs/1"
    assert len(selected_repos) == 1
    assert selected_repos[0]["url"] == "https://github.com/repo/a"


def test_summarizer_fallback_generates_chinese_source_grounded_summary(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    items = [
        {
            "title": "Example Paper",
            "url": "https://arxiv.org/abs/example",
            "topic": "medical image segmentation",
            "published": "2026-07-28T00:00:00Z",
            "authors": ["A", "B", "C", "D"],
            "venue_hint": "arXiv preprint",
        }
    ]

    summarized = summarize_selected_items(items, item_type="paper")

    assert "summary_cn" in summarized[0]
    assert "原始链接：https://arxiv.org/abs/example" in summarized[0]["summary_cn"]
    assert "具体贡献" in summarized[0]["summary_cn"]


def test_format_daily_brief_is_concise_chinese_and_keeps_links():
    content = format_daily_brief(
        brief_date="2026-07-28",
        days=7,
        topics=["medical image segmentation"],
        papers=[
            {
                "title": "Example Paper",
                "url": "https://arxiv.org/abs/example",
                "summary_cn": (
                    "这是一篇医学图像分割论文。它需要进一步核查实验设置。"
                    "原始链接：https://arxiv.org/abs/example"
                ),
            }
        ],
        repositories=[
            {
                "title": "repo/example",
                "url": "https://github.com/repo/example",
                "summary_cn": (
                    "这是一个医学图像分割项目。复用前需要检查许可证。"
                    "原始链接：https://github.com/repo/example"
                ),
            }
        ],
    )

    assert "# 每日 AI 科研简报" in content
    assert "## 论文精选" in content
    assert "https://arxiv.org/abs/example" in content
    assert "https://github.com/repo/example" in content
    assert "This paper proposes" not in content


def test_invoke_daily_brief_writes_selected_items_only(tmp_path, monkeypatch):
    import brief.workflow as workflow_module

    def fake_papers(topics, days, max_results_per_topic):
        return [
            {
                "title": "Paper A",
                "url": "https://arxiv.org/abs/a",
                "topic": topics[0],
                "published": "2026-07-28T00:00:00Z",
                "summary": "medical image segmentation " * 20,
                "authors": ["A"],
            },
            {
                "title": "Paper B",
                "url": "https://arxiv.org/abs/b",
                "topic": topics[0],
                "published": "2026-07-27T00:00:00Z",
                "summary": "short",
                "authors": ["B"],
            },
        ]

    def fake_repos(topics, days, max_results_per_topic):
        return [
            {
                "title": "repo/a",
                "url": "https://github.com/repo/a",
                "topic": topics[0],
                "updated_at": "2026-07-28T00:00:00Z",
                "summary": "toolkit",
                "stars": 10,
            }
        ]

    def fake_save(content, brief_date):
        output_path = tmp_path / f"{brief_date}.md"
        output_path.write_text(content, encoding="utf-8")
        return str(output_path)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(workflow_module, "collect_recent_arxiv_papers", fake_papers)
    monkeypatch.setattr(workflow_module, "collect_recent_github_repositories", fake_repos)
    monkeypatch.setattr(workflow_module, "save_daily_brief", fake_save)

    state = invoke_daily_brief(
        topics=["medical image segmentation"],
        days=7,
        brief_date="2026-07-28",
        selected_papers=1,
        selected_repos=1,
    )

    content = state["brief"]
    assert "Paper A" in content
    assert "Paper B" not in content
    assert "repo/a" in content
    assert "原始链接：https://arxiv.org/abs/a" in content
