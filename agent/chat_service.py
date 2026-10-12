"""Interactive chat service for the research assistant."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from agent.state import ResearchState
from agent.workflow import invoke, stream_updates
from memory.store import SQLiteMemoryStore
from rag.indexer import build_or_update_index
from rag.vector_store import vector_store_count


logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Formatted response plus raw LangGraph state."""

    answer: str
    state: ResearchState
    progress: list[dict] | None = None


@dataclass
class ChatProgressEvent:
    """User-facing progress event for streaming UI."""

    kind: Literal["progress", "final"]
    title: str
    detail: str
    state: ResearchState | None = None
    answer: str = ""


def ensure_knowledge_base() -> dict | None:
    """Build the local RAG index if it is empty."""

    if vector_store_count() == 0:
        return build_or_update_index()
    return None


def ask_agent(query: str) -> ChatResponse:
    """Run one user question through the LangGraph assistant."""

    initial_state = _initial_state(query)
    final_state = invoke(initial_state)
    _save_interaction_memory(final_state)
    return ChatResponse(answer=format_chat_answer(final_state), state=final_state)


def ask_agent_with_progress(query: str) -> Iterator[ChatProgressEvent]:
    """Run one question and yield public progress summaries as nodes finish."""

    yield ChatProgressEvent(
        kind="progress",
        title="接收问题",
        detail="已收到问题，开始进入规划、检索、工具调用、回答生成和反思检查流程。",
    )

    final_state: ResearchState | None = None
    progress_records: list[dict] = []
    for node_name, update, current_state in stream_updates(_initial_state(query)):
        title, detail = summarize_node_update(node_name, update, current_state)
        progress_records.append({"node": node_name, "title": title, "detail": detail})
        final_state = current_state
        yield ChatProgressEvent(kind="progress", title=title, detail=detail)

    if final_state is None:
        raise RuntimeError("Agent workflow did not produce a final state.")

    final_state["progress"] = progress_records
    _save_interaction_memory(final_state)
    yield ChatProgressEvent(
        kind="final",
        title="回答完成",
        detail="已生成最终回答，并保存本轮交互记忆。",
        state=final_state,
        answer=format_chat_answer(final_state),
    )


def format_chat_answer(state: ResearchState) -> str:
    """Format a concise chat answer from the final LangGraph state."""

    sections = [state.get("report", "").strip() or "Agent did not return a report."]

    reflection = state.get("reflection")
    report_error = state.get("report_error")
    if reflection or report_error:
        status = "通过" if reflection and reflection.get("passed") else "需要注意"
        quality_lines = ["### 质量检查", "", f"- 状态：{status}"]
        if report_error:
            quality_lines.append("- LLM：远程模型当前不可用，已使用本地 fallback。")
        for issue in (reflection or {}).get("issues", []):
            quality_lines.append(f"- {issue}")
        sections.append("\n".join(quality_lines))

    memory_records = state.get("memory_records", [])
    if memory_records:
        sections.append(f"### 历史记忆\n\n- 命中 {len(memory_records)} 条历史交互记忆。")

    trace = state.get("reasoning_trace", [])
    if trace:
        trace_lines = ["### 工具调用轨迹"]
        for item in trace:
            trace_lines.append(f"- {item.get('action')}: {item.get('observation')}")
        sections.append("\n".join(trace_lines))

    return "\n\n".join(section for section in sections if section)


def summarize_node_update(
    node_name: str,
    update: dict,
    state: ResearchState,
) -> tuple[str, str]:
    """Create a public, compact summary of one workflow node update."""

    if node_name == "planner_node":
        tools = ", ".join(update.get("selected_tools", [])) or "本地 RAG"
        return (
            "规划完成",
            f"任务类型：{update.get('task_type', 'research')}；"
            f"计划步骤：{len(update.get('plan', []))}；工具策略：{tools}。",
        )

    if node_name == "memory_node":
        docs = update.get("retrieved_documents", [])
        records = update.get("memory_records", [])
        detail = f"召回 {len(docs)} 个知识片段，命中 {len(records)} 条历史交互记忆。"
        if docs:
            first = docs[0]
            source = first.get("filename") or first.get("source", "")
            page = first.get("page_start")
            if page:
                detail += f" 首条来源：{source} p.{page}。"
            elif source:
                detail += f" 首条来源：{source}。"
        return "检索完成", detail

    if node_name == "research_node":
        trace = update.get("reasoning_trace", [])
        results = update.get("research_results", [])
        actions = ", ".join(item.get("action", "") for item in trace if item.get("action"))
        return (
            "工具调用完成",
            f"执行动作：{actions or '跳过外部工具'}；获得 {len(results)} 条外部结果。",
        )

    if node_name == "daily_brief_node":
        results = update.get("research_results", [])
        output_path = results[0].get("output_path") if results else ""
        return "日报生成完成", f"每日简报已生成：{output_path or '未返回文件路径'}。"

    if node_name == "report_node":
        error = update.get("report_error")
        context_summary = update.get("context_summary", {})
        context_detail = ""
        if context_summary:
            context_detail = (
                f"上下文包：{context_summary.get('packet_count', 0)}；"
                f"估算 tokens：{context_summary.get('estimated_tokens', 0)}/"
                f"{context_summary.get('budget_tokens', 0)}。"
            )
        if error:
            return (
                "回答生成完成，但远程 LLM 不可用",
                "已使用本地 fallback 基于召回材料生成回答，请优先检查模型名和中转服务可用性。"
                + context_detail,
            )
        return (
            "回答生成完成",
            "已基于检索材料、工具结果和历史记忆生成回答。" + context_detail,
        )

    if node_name == "reflection_node":
        reflection = update.get("reflection", {})
        issues = reflection.get("issues", [])
        if reflection.get("passed"):
            return (
                "反思检查通过",
                "来源、检索记忆和不确定性表达通过基础检查。",
            )
        return "反思检查提示", f"发现 {len(issues)} 个需要注意的问题：" + "；".join(issues[:2])

    return node_name, "节点已完成。"


def _initial_state(query: str) -> ResearchState:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Query cannot be empty.")

    return {
        "query": normalized_query,
        "memory": "",
        "memory_records": [],
        "retrieved_documents": [],
        "research_results": [],
        "report": "",
    }


def _save_interaction_memory(final_state: ResearchState) -> None:
    try:
        SQLiteMemoryStore().add_interaction(dict(final_state))
    except Exception as exc:  # pragma: no cover - memory must not break chat UX.
        logger.warning("Failed to persist interaction memory: %s", exc)
