"""Evaluate selective domain gating for ID/OOD research questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import evaluate_retrieval_case
from rag.retriever import retrieve_documents


DEFAULT_CASES = Path("data/eval/rag_eval_cases_v1.jsonl")
DEFAULT_THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate confidence-based ID/OOD rejection.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL retrieval cases with OOD negatives.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--thresholds", default=",".join(str(item) for item in DEFAULT_THRESHOLDS))
    parser.add_argument("--output", default="", help="Optional UTF-8 JSON output path.")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    precomputed = [_run_case(case, args.top_k) for case in cases]
    curves = [_score_threshold(precomputed, threshold) for threshold in thresholds]
    payload = json.dumps(
        {
            "case_count": len(cases),
            "top_k": args.top_k,
            "thresholds": thresholds,
            "curves": curves,
            "recommended": _recommend(curves),
        },
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _run_case(case: dict[str, Any], top_k: int) -> dict[str, Any]:
    retrieved = retrieve_documents(
        case["question"],
        top_k=top_k,
        expand=False,
        document_types=case.get("document_types"),
        enable_dense=True,
        enable_bm25=True,
        domain_guard=False,
        candidate_k=20,
        rerank=True,
    )
    metrics = evaluate_retrieval_case(case, retrieved, k_values=(5,))
    max_score = max((float(item.get("memory_score", item.get("score", 0.0))) for item in retrieved), default=0.0)
    is_ood = bool(case.get("no_relevant_expected")) or (
        not case.get("expected_sources") and not case.get("expected_keywords") and not case.get("expected_pages")
    )
    return {
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "is_ood": is_ood,
        "max_score": max_score,
        "hit@5": float(metrics.get("hit@5", 0.0)),
        "retrieved_count": len(retrieved),
    }


def _score_threshold(cases: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    id_cases = [case for case in cases if not case["is_ood"]]
    ood_cases = [case for case in cases if case["is_ood"]]
    accepted_id = [case for case in id_cases if case["max_score"] >= threshold]
    accepted_ood = [case for case in ood_cases if case["max_score"] >= threshold]
    id_hits = [case for case in accepted_id if case["hit@5"] > 0]
    rejected_ood = [case for case in ood_cases if case["max_score"] < threshold]
    return {
        "threshold": threshold,
        "coverage": _ratio(len(accepted_id) + len(accepted_ood), len(cases)),
        "id_recall@5": _ratio(len(id_hits), len(id_cases)),
        "id_accept_rate": _ratio(len(accepted_id), len(id_cases)),
        "ood_false_accept_rate": _ratio(len(accepted_ood), len(ood_cases)),
        "refusal_accuracy": _ratio(len(rejected_ood), len(ood_cases)),
    }


def _recommend(curves: list[dict[str, Any]]) -> dict[str, Any]:
    if not curves:
        return {}
    viable = [curve for curve in curves if float(curve["id_recall@5"]) >= 0.45]
    if not viable:
        viable = curves
    return min(viable, key=lambda curve: (float(curve["ood_false_accept_rate"]), -float(curve["id_recall@5"])))


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


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


if __name__ == "__main__":
    main()
