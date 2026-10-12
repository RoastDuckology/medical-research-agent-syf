"""Evaluate agent routing, tool selection, and fallback behavior."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent.workflow import invoke
from agent.nodes.planner_node import planner_node


DEFAULT_CASES = Path("data/eval/agent_routing_cases_v1.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent workflow routing.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL workflow evaluation cases.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    parser.add_argument(
        "--planner-only",
        action="store_true",
        help="Evaluate routing/tool selection without executing external tools.",
    )
    parser.add_argument("--output", default="", help="Optional UTF-8 JSON output path.")
    parser.add_argument(
        "--recorded-tools",
        action="store_true",
        help="Replay recorded arXiv/GitHub responses instead of live network calls.",
    )
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    if args.limit > 0:
        cases = cases[: args.limit]

    with _temporary_env(RESEARCH_TOOL_MODE="recorded") if args.recorded_tools else _null_context():
        results = [
            _evaluate_planner_case(case) if args.planner_only else _evaluate_case(case)
            for case in cases
        ]
    payload = json.dumps(
        {
            "case_count": len(results),
            "aggregate": _aggregate(results),
            "cases": results,
        },
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    expected_task_type = case.get("expected_task_type")
    expected_tools = list(case.get("expected_tools", []))
    expected_fallback = bool(case.get("expected_fallback", False))
    try:
        state = invoke(
            {
                "query": case["query"],
                "memory": "",
                "memory_records": [],
                "retrieved_documents": [],
                "research_results": [],
                "report": "",
            }
        )
        selected_tools = list(state.get("selected_tools", []))
        reasoning_trace = list(state.get("reasoning_trace", []))
        tool_actions = [item.get("action") for item in reasoning_trace]
        report_error = state.get("report_error", "")
        return {
            "query": case["query"],
            "expected_task_type": expected_task_type,
            "actual_task_type": state.get("task_type", ""),
            "task_routing_correct": _bool_score(state.get("task_type") == expected_task_type),
            "expected_tools": expected_tools,
            "selected_tools": selected_tools,
            "tool_selection_correct": _bool_score(_same_tools(selected_tools, expected_tools)),
            "tool_call_success": _bool_score(_tool_call_success(tool_actions, expected_tools, state)),
            "task_completion": _bool_score(bool(state.get("report")) or bool(state.get("research_results"))),
            "fallback_success": _bool_score((not expected_fallback) or bool(report_error)),
            "report_error": report_error,
            "latency_seconds": time.perf_counter() - started_at,
        }
    except Exception as exc:
        return {
            "query": case["query"],
            "expected_task_type": expected_task_type,
            "actual_task_type": "",
            "task_routing_correct": 0.0,
            "expected_tools": expected_tools,
            "selected_tools": [],
            "tool_selection_correct": 0.0,
            "tool_call_success": 0.0,
            "task_completion": 0.0,
            "fallback_success": 0.0,
            "report_error": str(exc),
            "latency_seconds": time.perf_counter() - started_at,
        }


def _evaluate_planner_case(case: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    expected_task_type = case.get("expected_task_type")
    expected_tools = list(case.get("expected_tools", []))
    update = planner_node({"query": case["query"]})
    selected_tools = list(update.get("selected_tools", []))
    return {
        "query": case["query"],
        "expected_task_type": expected_task_type,
        "actual_task_type": update.get("task_type", ""),
        "task_routing_correct": _bool_score(update.get("task_type") == expected_task_type),
        "expected_tools": expected_tools,
        "selected_tools": selected_tools,
        "tool_selection_correct": _bool_score(_same_tools(selected_tools, expected_tools)),
        "tool_call_success": 1.0,
        "task_completion": 1.0,
        "fallback_success": 1.0,
        "report_error": "",
        "latency_seconds": time.perf_counter() - started_at,
    }
def _same_tools(actual: list[str], expected: list[str]) -> bool:
    return sorted(actual) == sorted(expected)


def _tool_call_success(actions: list[str], expected_tools: list[str], state: dict[str, Any]) -> bool:
    if not expected_tools:
        return "local_rag_only" in actions or not state.get("research_results")
    if expected_tools == ["daily_brief"]:
        return bool(state.get("research_results"))
    return all(tool in actions for tool in expected_tools)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "task_routing_correct",
        "tool_selection_correct",
        "tool_call_success",
        "task_completion",
        "fallback_success",
        "latency_seconds",
    ]
    if not results:
        return {}
    return {
        key: sum(float(result.get(key, 0.0)) for result in results) / len(results)
        for key in keys
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Workflow evaluation case file not found: {path}")
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        case = json.loads(stripped)
        if "query" not in case:
            raise ValueError(f"Missing query at {path}:{line_number}")
        cases.append(case)
    return cases


def _bool_score(value: bool) -> float:
    return 1.0 if value else 0.0


@contextmanager
def _temporary_env(**updates: str):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _null_context():
    yield


if __name__ == "__main__":
    main()
