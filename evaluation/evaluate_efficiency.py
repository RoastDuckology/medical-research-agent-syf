"""Measure latency and context-budget efficiency for agent runs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from agent.chat_service import ask_agent


DEFAULT_CASES = Path("data/eval/qa_eval_cases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent latency and context efficiency.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL QA or workflow cases.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    parser.add_argument("--output", default="", help="Optional UTF-8 JSON output path.")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    if args.limit > 0:
        cases = cases[: args.limit]
    results = [_run_case(case) for case in cases]
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


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    query = case.get("query") or case.get("question")
    started_at = time.perf_counter()
    try:
        response = ask_agent(query)
        state = response.state
        trace = state.get("reasoning_trace", [])
        context_summary = state.get("context_summary", {})
        return {
            "query": query,
            "latency_seconds": time.perf_counter() - started_at,
            "llm_call_count": 1,
            "tool_call_count": len([item for item in trace if item.get("action") not in {"local_rag_only", "skip"}]),
            "context_estimated_tokens": int(context_summary.get("estimated_tokens", 0)),
            "context_budget_tokens": int(context_summary.get("budget_tokens", 0)),
            "context_packet_count": int(context_summary.get("packet_count", 0)),
            "fallback_used": 1.0 if state.get("report_error") else 0.0,
            "report_error": state.get("report_error", ""),
        }
    except Exception as exc:
        return {
            "query": query,
            "latency_seconds": time.perf_counter() - started_at,
            "llm_call_count": 0,
            "tool_call_count": 0,
            "context_estimated_tokens": 0,
            "context_budget_tokens": 0,
            "context_packet_count": 0,
            "fallback_used": 0.0,
            "report_error": str(exc),
        }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    latencies = sorted(float(result.get("latency_seconds", 0.0)) for result in results)
    return {
        "p50_latency_seconds": _percentile(latencies, 50),
        "p95_latency_seconds": _percentile(latencies, 95),
        "mean_latency_seconds": statistics.mean(latencies),
        "mean_llm_call_count": _mean(results, "llm_call_count"),
        "mean_tool_call_count": _mean(results, "tool_call_count"),
        "mean_context_estimated_tokens": _mean(results, "context_estimated_tokens"),
        "mean_context_packet_count": _mean(results, "context_packet_count"),
        "fallback_rate": _mean(results, "fallback_used"),
    }


def _percentile(sorted_values: list[float], percentile: int) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round((percentile / 100) * (len(sorted_values) - 1))))
    return sorted_values[index]


def _mean(results: list[dict[str, Any]], key: str) -> float:
    return sum(float(result.get(key, 0.0)) for result in results) / len(results)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        case = json.loads(stripped)
        if "query" not in case and "question" not in case:
            raise ValueError(f"Missing query/question at {path}:{line_number}")
        cases.append(case)
    return cases


if __name__ == "__main__":
    main()
