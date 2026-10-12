"""Plan-and-Solve style planning node for research queries."""

from __future__ import annotations

from agent.state import ResearchState


PAPER_KEYWORDS = {
    "paper",
    "papers",
    "arxiv",
    "literature",
    "survey",
    "论文",
    "文献",
    "综述",
    "顶会",
    "顶刊",
}
CODE_KEYWORDS = {
    "github",
    "repo",
    "repository",
    "code",
    "implementation",
    "开源",
    "代码",
    "仓库",
}
BRIEF_KEYWORDS = {"daily", "brief", "日报", "简报", "每日"}
LOCAL_ONLY_KEYWORDS = {
    "pdf",
    "data/papers",
    "paper_text",
    "实验进度",
    "实验记录",
    "本地",
    "知识库",
    "source link",
    "project use",
}


def planner_node(state: ResearchState) -> dict:
    """Create a lightweight plan, task type, and likely useful tools."""

    query = state.get("query", "").strip()
    task_type = _classify_task(query)
    selected_tools = _select_tools(query, task_type)
    plan = _build_plan(task_type, selected_tools)
    return {
        "task_type": task_type,
        "plan": plan,
        "selected_tools": selected_tools,
    }


def _classify_task(query: str) -> str:
    normalized = query.lower()
    if _is_local_rag_question(normalized):
        return "research"
    if _contains_any(normalized, BRIEF_KEYWORDS):
        return "daily_brief"
    return "research"


def _select_tools(query: str, task_type: str) -> list[str]:
    normalized = query.lower()
    if task_type == "daily_brief":
        return ["daily_brief"]
    if _is_local_rag_question(normalized):
        return []

    tools: list[str] = []
    if "arxiv" in normalized or _contains_any(normalized, {"查找论文", "搜索论文", "找论文", "相关论文"}):
        tools.append("arxiv")
    elif _contains_any(normalized, PAPER_KEYWORDS) and _contains_any(normalized, {"查找", "搜索", "find", "search"}):
        tools.append("arxiv")

    if _contains_any(normalized, CODE_KEYWORDS):
        tools.append("github")

    ordered = ["arxiv", "github"]
    return [tool for tool in ordered if tool in set(tools)]


def _build_plan(task_type: str, selected_tools: list[str]) -> list[dict]:
    if task_type == "daily_brief":
        return [
            {
                "step": "generate_daily_brief",
                "goal": "Generate a concise Chinese daily AI brief with source links.",
            },
            {
                "step": "reflect",
                "goal": "Check whether the brief was generated and whether source links are preserved.",
            },
        ]

    steps = [
        {
            "step": "retrieve_memory",
            "goal": "Retrieve source-grounded local memory, paper PDFs, and experiment notes.",
        },
    ]
    if selected_tools:
        steps.append(
            {
                "step": "external_research",
                "goal": f"Use selected tools only: {', '.join(selected_tools)}.",
            }
        )
    else:
        steps.append(
            {
                "step": "local_only",
                "goal": "Skip external tools and answer from local RAG evidence.",
            }
        )
    steps.extend(
        [
            {
                "step": "synthesize_report",
                "goal": "Generate a concise source-aware research answer.",
            },
            {
                "step": "reflect",
                "goal": "Check whether the report preserves sources and uncertainty.",
            },
        ]
    )
    return steps


def _is_local_rag_question(normalized_query: str) -> bool:
    if _contains_any(normalized_query, LOCAL_ONLY_KEYWORDS):
        return True
    if "in the " in normalized_query and " paper" in normalized_query:
        return True
    if normalized_query.startswith(("what is the source", "summarize current", "总结当前")):
        return True
    return False


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)
