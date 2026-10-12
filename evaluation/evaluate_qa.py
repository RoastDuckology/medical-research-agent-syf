"""Evaluate end-to-end QA behavior of the agent.

Usage:
    python -m evaluation.evaluate_qa
    python -m evaluation.evaluate_qa --cases data/eval/qa_eval_cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agent.chat_service import ask_agent


DEFAULT_CASES = Path("data/eval/qa_eval_cases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end QA behavior.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL QA evaluation cases.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    parser.add_argument(
        "--skip-llm-errors",
        action="store_true",
        help="Keep running and mark LLM/provider failures as failed cases.",
    )
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    if args.limit > 0:
        cases = cases[: args.limit]
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"Running QA case {index}/{len(cases)}: {case['question']}", flush=True)
        results.append(_evaluate_case(case))
    aggregate = _aggregate(results)
    print(
        json.dumps(
            {
                "case_count": len(results),
                "aggregate": aggregate,
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        response = ask_agent(case["question"])
    except Exception as exc:
        return {
            "question": case["question"],
            "keyword_recall": 0.0,
            "citation_coverage": 0.0,
            "retrieval_hit": 0.0,
            "reflection_pass": 0.0,
            "answer_chars": 0,
            "latency_seconds": time.perf_counter() - started_at,
            "report_error": str(exc),
        }
    answer = response.answer
    state = response.state
    expected_keywords = [str(item) for item in case.get("expected_keywords", [])]
    required_citations = [str(item) for item in case.get("required_citations", [])]

    keyword_hits = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in answer.lower()
    ]
    citation_hits = [
        citation
        for citation in required_citations
        if citation.lower() in answer.lower()
        or citation.lower() in _source_blob(state).lower()
    ]
    retrieved_documents = state.get("retrieved_documents", [])
    reflection = state.get("reflection", {})

    return {
        "question": case["question"],
        "keyword_recall": len(keyword_hits) / len(expected_keywords) if expected_keywords else 1.0,
        "citation_coverage": len(citation_hits) / len(required_citations) if required_citations else 1.0,
        "retrieval_hit": 1.0 if retrieved_documents else 0.0,
        "reflection_pass": 1.0 if reflection.get("passed") else 0.0,
        "answer_chars": len(answer),
        "latency_seconds": time.perf_counter() - started_at,
        "report_error": state.get("report_error", ""),
    }


def _source_blob(state: dict[str, Any]) -> str:
    parts: list[str] = []
    for document in state.get("retrieved_documents", []) or []:
        parts.extend(
            str(document.get(key, ""))
            for key in ("source", "filename", "abs_url", "pdf_url", "heading_path")
        )
    for result in state.get("research_results", []) or []:
        parts.extend(str(value) for value in result.values())
    return " ".join(parts)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    keys = [
        "keyword_recall",
        "citation_coverage",
        "retrieval_hit",
        "reflection_pass",
        "answer_chars",
        "latency_seconds",
    ]
    return {
        key: sum(float(result.get(key, 0.0)) for result in results) / len(results)
        for key in keys
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        case = json.loads(stripped)
        if "question" not in case:
            raise ValueError(f"Missing question at {path}:{line_number}")
        cases.append(case)
    return cases


if __name__ == "__main__":
    main()
