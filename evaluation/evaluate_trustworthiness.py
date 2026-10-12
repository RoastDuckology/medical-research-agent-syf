"""Evaluate answer trustworthiness from fixed QA annotations."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from agent.chat_service import ask_agent


DEFAULT_CASES = Path("data/eval/trustworthiness_cases_v1.jsonl")
SOURCE_MARKERS = ("http://", "https://", "file:", "page:", "heading_path:", "Source:")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate source-grounded answer behavior.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL trustworthiness cases.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    parser.add_argument("--output", default="", help="Optional UTF-8 JSON output path.")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    if args.limit > 0:
        cases = cases[: args.limit]
    results = [_evaluate_case(case) for case in cases]
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
    try:
        response = ask_agent(case["query"])
    except Exception as exc:
        return {
            "query": case["query"],
            "citation_coverage": 0.0,
            "citation_precision": 0.0,
            "source_location_accuracy": 0.0,
            "grounded_answer_rate": 0.0,
            "hallucination_rate": 1.0,
            "latency_seconds": time.perf_counter() - started_at,
            "error": str(exc),
        }

    answer = response.answer
    evidence_blob = _evidence_blob(response.state)
    claims = [str(item) for item in case.get("required_claims", [])]
    citations = [str(item) for item in case.get("required_citations", [])]
    locations = [str(item) for item in case.get("required_locations", [])]

    grounded_claims = [claim for claim in claims if _contains(evidence_blob, claim)]
    answer_claims = [claim for claim in claims if _contains(answer, claim)]
    cited_claims = [claim for claim in answer_claims if _claim_line_has_source(answer, claim)]
    citation_hits = [citation for citation in citations if _contains(answer + "\n" + evidence_blob, citation)]
    location_hits = [location for location in locations if _contains(answer + "\n" + evidence_blob, location)]

    unsupported_answer_claims = [claim for claim in answer_claims if claim not in grounded_claims]
    return {
        "query": case["query"],
        "citation_coverage": _ratio(len(cited_claims), len(claims)),
        "citation_precision": _ratio(len(citation_hits), len(citations)),
        "source_location_accuracy": _ratio(len(location_hits), len(locations)),
        "grounded_answer_rate": _ratio(len(grounded_claims), len(claims)),
        "hallucination_rate": _ratio(len(unsupported_answer_claims), len(answer_claims)),
        "latency_seconds": time.perf_counter() - started_at,
        "report_error": response.state.get("report_error", ""),
    }


def _evidence_blob(state: dict[str, Any]) -> str:
    parts: list[str] = []
    for document in state.get("retrieved_documents", []) or []:
        parts.extend(str(value) for value in document.values())
    for result in state.get("research_results", []) or []:
        parts.extend(str(value) for value in result.values())
    parts.append(str(state.get("report", "")))
    return "\n".join(parts)


def _claim_line_has_source(answer: str, claim: str) -> bool:
    for line in answer.splitlines():
        if _contains(line, claim) and any(marker in line for marker in SOURCE_MARKERS):
            return True
    return False


def _contains(text: str, needle: str) -> bool:
    if not needle:
        return False
    return needle.lower() in text.lower()


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "citation_coverage",
        "citation_precision",
        "source_location_accuracy",
        "grounded_answer_rate",
        "hallucination_rate",
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
        raise FileNotFoundError(f"Trustworthiness case file not found: {path}")
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


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator
