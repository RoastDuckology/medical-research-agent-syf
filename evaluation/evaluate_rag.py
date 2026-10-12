"""Run RAG retrieval benchmarks and ablations.

Usage:
    python -m evaluation.evaluate_rag
    python -m evaluation.evaluate_rag --mode all --cases data/eval/rag_eval_cases_v1.jsonl --top-k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import aggregate_metrics, evaluate_retrieval_case
from rag.retriever import retrieve_documents


DEFAULT_CASES = Path("data/eval/rag_eval_cases.jsonl")

RETRIEVAL_MODES = {
    "baseline": {"enable_dense": True, "enable_bm25": False, "expand": False, "rerank": False, "candidate_k": 20},
    "hybrid": {"enable_dense": True, "enable_bm25": True, "expand": False, "rerank": False, "candidate_k": 20},
    "hybrid_rerank": {"enable_dense": True, "enable_bm25": True, "expand": False, "rerank": True, "candidate_k": 20},
    "full": {"enable_dense": True, "enable_bm25": True, "expand": True, "rerank": True, "candidate_k": 20},
    "adaptive": {"enable_dense": True, "enable_bm25": True, "expand": "adaptive", "rerank": True, "candidate_k": 20},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval metrics.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL evaluation cases.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum retrieved chunks per question.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "hybrid", "hybrid_rerank", "full", "adaptive", "all"],
        default="all",
        help="Retrieval mode: baseline=dense, hybrid=dense+BM25, full=dense+BM25+MQE.",
    )
    parser.add_argument("--no-expand", action="store_true", help="Compatibility flag: disable MQE for selected mode.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Optional retrieval score threshold.")
    parser.add_argument("--disable-domain-guard", action="store_true", help="Disable corpus-domain guard.")
    parser.add_argument("--output", default="", help="Optional UTF-8 JSON output path.")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    modes = list(RETRIEVAL_MODES) if args.mode == "all" else [args.mode]
    reports = {
        mode: _evaluate_mode(
            mode,
            cases,
            top_k=args.top_k,
            force_no_expand=args.no_expand,
            domain_guard=not args.disable_domain_guard,
        )
        if args.min_score == 0
        else _evaluate_mode(
            mode,
            cases,
            top_k=args.top_k,
            force_no_expand=args.no_expand,
            min_score=args.min_score,
            domain_guard=not args.disable_domain_guard,
        )
        for mode in modes
    }
    output = {
        "case_count": len(cases),
        "top_k": args.top_k,
        "min_score": args.min_score,
        "domain_guard": not args.disable_domain_guard,
        "modes": reports,
        "comparison": _compare_modes(reports),
    }
    payload = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _evaluate_mode(
    mode: str,
    cases: list[dict[str, Any]],
    top_k: int,
    force_no_expand: bool = False,
    min_score: float = 0.0,
    domain_guard: bool = True,
) -> dict[str, Any]:
    config = dict(RETRIEVAL_MODES[mode])
    if force_no_expand:
        config["expand"] = False
    case_metrics: list[dict[str, Any]] = []
    for case in cases:
        retrieved = retrieve_documents(
            case["question"],
            top_k=top_k,
            expand=_should_expand(case["question"], config["expand"]),
            document_types=case.get("document_types"),
            enable_dense=bool(config["enable_dense"]),
            enable_bm25=bool(config["enable_bm25"]),
            min_score=min_score,
            domain_guard=domain_guard,
            candidate_k=int(config["candidate_k"]),
            rerank=bool(config["rerank"]),
        )
        metrics = evaluate_retrieval_case(
            case,
            retrieved,
            k_values=tuple(sorted(set(k for k in (3, 5, top_k) if k <= top_k))),
        )
        metrics["mode"] = mode
        case_metrics.append(metrics)

    return {
        "config": config,
        "aggregate": aggregate_metrics(case_metrics),
        "cases": case_metrics,
    }


def _compare_modes(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    baseline = reports.get("baseline", {}).get("aggregate", {})
    full = reports.get("full", {}).get("aggregate", {})
    hybrid = reports.get("hybrid", {}).get("aggregate", {})
    hybrid_rerank = reports.get("hybrid_rerank", {}).get("aggregate", {})
    adaptive = reports.get("adaptive", {}).get("aggregate", {})
    for metric in ("recall@3", "recall@5", "hit@5", "mrr", "pdf_page_hit", "false_positive"):
        if metric in baseline and metric in full:
            comparison[f"full_vs_baseline_{metric}_pp"] = round(
                (float(full[metric]) - float(baseline[metric])) * 100,
                2,
            )
        if metric in baseline and metric in hybrid:
            comparison[f"hybrid_vs_baseline_{metric}_pp"] = round(
                (float(hybrid[metric]) - float(baseline[metric])) * 100,
                2,
            )
        if metric in hybrid and metric in hybrid_rerank:
            comparison[f"hybrid_rerank_vs_hybrid_{metric}_pp"] = round(
                (float(hybrid_rerank[metric]) - float(hybrid[metric])) * 100,
                2,
            )
        if metric in baseline and metric in adaptive:
            comparison[f"adaptive_vs_baseline_{metric}_pp"] = round(
                (float(adaptive[metric]) - float(baseline[metric])) * 100,
                2,
            )
    return comparison


def _should_expand(question: str, expand_setting: bool | str) -> bool:
    if expand_setting is True:
        return True
    if expand_setting is False:
        return False
    normalized = question.lower()
    complex_markers = {
        "compare",
        "cross",
        "recent",
        "latest",
        "arxiv",
        "github",
        "综合",
        "比较",
        "近",
        "最新",
        "跨论文",
        "相关",
    }
    return any(marker in normalized for marker in complex_markers) or len(normalized.split()) >= 14


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation case file not found: {path}")

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
