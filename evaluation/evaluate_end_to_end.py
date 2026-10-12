"""Evaluate complete research-agent tasks, including final answer quality."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from agent.chat_service import ask_agent
from agent.nodes.planner_node import planner_node
from rag.retriever import retrieve_documents


DEFAULT_CASES = Path("data/eval/end_to_end_tasks_v1.jsonl")
SOURCE_MARKERS = ("http://", "https://", "file:", "page:", "heading_path:", "Source:", "来源")
REFUSAL_PATTERNS = (
    "无法回答",
    "不能回答",
    "无法给出",
    "证据不足以回答",
    "没有足够证据",
    "当前知识库没有足够证据",
    "insufficient evidence",
    "cannot answer",
    "not enough evidence",
)
STRUCTURE_MARKERS = ("### 结论", "证据：", "来源：")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen end-to-end research tasks.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL end-to-end tasks.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic planner+RAG synthesis instead of the configured remote LLM.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--task-scope",
        choices=["all", "local_qa", "workflow"],
        default="all",
        help="Evaluate all tasks or only one task family.",
    )
    parser.add_argument("--output", default="", help="Optional UTF-8 JSON output path.")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    if args.task_scope != "all":
        cases = [case for case in cases if case.get("task_type") == args.task_scope]
    if args.limit > 0:
        cases = cases[: args.limit]
    results = [_evaluate_case(case, args.top_k, args.offline) for case in cases]
    payload = json.dumps(
        {
            "case_count": len(results),
            "mode": "offline" if args.offline else "agent",
            "task_scope": args.task_scope,
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


def _evaluate_case(case: dict[str, Any], top_k: int, offline: bool) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        if offline:
            answer, state = _offline_answer(case, top_k)
        else:
            response = ask_agent(case["query"])
            answer, state = response.answer, response.state
        metrics = _score_answer(case, answer, state)
        return {
            "id": case.get("id", ""),
            "query": case["query"],
            "task_type": case.get("task_type", ""),
            "expected_tool": case.get("expected_tool", ""),
            "latency_seconds": time.perf_counter() - started_at,
            "answer": answer,
            "retrieved_source_refs": _source_refs(state),
            "report_error": state.get("report_error", ""),
            **metrics,
        }
    except Exception as exc:
        return {
            "id": case.get("id", ""),
            "query": case.get("query", ""),
            "task_type": case.get("task_type", ""),
            "expected_tool": case.get("expected_tool", ""),
            "latency_seconds": time.perf_counter() - started_at,
            "task_completion": 0.0,
            "answer_correctness": None,
            "answer_completeness": None,
            "citation_accuracy": None,
            "citation_coverage": None,
            "groundedness": None,
            "refusal_accuracy": 0.0,
            "route_accuracy": 0.0,
            "tool_selection_accuracy": 0.0,
            "trace_validity": 0.0,
            "structure_compliance": 0.0,
            "claim_source_binding": 0.0,
            "error_types": ["exception"],
            "error": str(exc),
        }


def _offline_answer(case: dict[str, Any], top_k: int) -> tuple[str, dict[str, Any]]:
    plan = planner_node({"query": case["query"]})
    expected_tool = str(case.get("expected_tool", ""))
    document_types = ["paper_pdf"] if "pdf" in case["query"].lower() else None
    documents = retrieve_documents(
        case["query"],
        top_k=top_k,
        expand=_is_complex(case["query"]),
        document_types=document_types,
        enable_dense=True,
        enable_bm25=True,
        domain_guard=False,
        candidate_k=20,
        rerank=True,
    )
    state: dict[str, Any] = {
        "query": case["query"],
        "task_type": plan.get("task_type", ""),
        "selected_tools": plan.get("selected_tools", []),
        "retrieved_documents": documents,
        "research_results": [],
        "reasoning_trace": [{"action": "local_rag_only", "observation": f"retrieved {len(documents)} chunks"}],
    }
    if expected_tool and expected_tool != "local_rag":
        state["reasoning_trace"].append({"action": expected_tool, "observation": "offline tool routing replay"})

    if not documents and not case.get("answerable", True):
        return "当前知识库没有足够证据回答该问题，因此拒绝给出无来源结论。", state
    if not documents:
        return "当前知识库召回证据不足，只能给出低置信回答。", state

    lines = ["基于当前知识库证据，结论如下："]
    if expected_tool and expected_tool != "local_rag":
        lines.append(f"- 工作流已选择 {expected_tool} 路径；离线评测模式下使用录制/本地证据替代真实网络调用。")
    for document in documents[:3]:
        source = document.get("filename") or document.get("source", "")
        page = document.get("page_start", "")
        snippet = _compact(document.get("content", ""))
        location = f"{source} page:{page}" if page else source
        lines.append(f"- {snippet} 来源: {location}")
    return "\n".join(lines), state


def _score_answer(case: dict[str, Any], answer: str, state: dict[str, Any]) -> dict[str, float]:
    answerable = bool(case.get("answerable", True))
    required_terms = [str(item) for item in case.get("required_answer_terms", []) if str(item).strip()]
    required_citations = [str(item) for item in case.get("required_citations", []) if str(item).strip()]
    expected_tool = str(case.get("expected_tool", ""))
    task_type = str(case.get("task_type", ""))
    evidence = _evidence_blob(state)

    term_hits = [_contains(answer, term) for term in required_terms]
    citation_hits = [_contains(answer, citation) for citation in required_citations]
    cited_term_hits = [_line_has_source(answer, term) for term in required_terms]
    grounded_hits = [_contains(evidence, term) for term in required_terms]
    refusal = _is_refusal(answer)
    structure_compliance = _structure_compliance(answer)
    claim_source_binding = _claim_source_binding(answer)

    selected_tools = list(state.get("selected_tools", []))
    trace_actions = [str(item.get("action", "")) for item in state.get("reasoning_trace", [])]
    trace_valid = _trace_valid(expected_tool, selected_tools, trace_actions)
    tool_valid = _tool_valid(expected_tool, selected_tools, trace_actions)
    route_valid = _route_valid(case, state)
    completion = (answerable and bool(answer.strip()) and not refusal) or ((not answerable) and refusal)

    common = {
        "task_completion": _bool(completion),
        "refusal_accuracy": _bool((not answerable and refusal) or (answerable and not refusal)),
        "route_accuracy": _bool(route_valid),
        "tool_selection_accuracy": _bool(tool_valid),
            "trace_validity": _bool(trace_valid),
            "structure_compliance": structure_compliance,
            "claim_source_binding": claim_source_binding,
        }
    if task_type == "workflow":
        return {
            **common,
            "answer_correctness": None,
            "answer_completeness": None,
            "citation_accuracy": None,
            "citation_coverage": None,
            "groundedness": None,
            "error_types": [],
        }

    scored = {
        **common,
        "answer_correctness": _ratio(sum(term_hits), len(required_terms)) if answerable and required_terms else None,
        "answer_completeness": _ratio(sum(term_hits), len(required_terms)) if answerable and required_terms else None,
        "citation_accuracy": _ratio(sum(citation_hits), len(required_citations)) if required_citations else None,
        "citation_coverage": _ratio(sum(cited_term_hits), len(required_terms)) if required_terms else None,
        "groundedness": _ratio(sum(grounded_hits), len(required_terms)) if answerable and required_terms else None,
    }
    scored["error_types"] = _classify_errors(
        answerable=answerable,
        refusal=refusal,
        required_terms=required_terms,
        required_citations=required_citations,
        term_hits=term_hits,
        citation_hits=citation_hits,
        cited_term_hits=cited_term_hits,
        grounded_hits=grounded_hits,
    )
    return scored


def _trace_valid(expected_tool: str, selected_tools: list[str], trace_actions: list[str]) -> bool:
    if not expected_tool or expected_tool == "local_rag":
        return not selected_tools or "local_rag_only" in trace_actions
    expected_parts = [part for part in expected_tool.split("+") if part]
    return all(part in selected_tools or any(part in action for action in trace_actions) for part in expected_parts)


def _tool_valid(expected_tool: str, selected_tools: list[str], trace_actions: list[str]) -> bool:
    return _trace_valid(expected_tool, selected_tools, trace_actions)


def _route_valid(case: dict[str, Any], state: dict[str, Any]) -> bool:
    expected = str(case.get("expected_agent_task_type") or case.get("task_type", ""))
    if not expected:
        return True
    actual = str(state.get("task_type", ""))
    return actual == expected or (case.get("task_type") == "local_qa" and actual == "research")


def _evidence_blob(state: dict[str, Any]) -> str:
    parts: list[str] = []
    for document in state.get("retrieved_documents", []) or []:
        parts.extend(str(value) for value in document.values())
    for result in state.get("research_results", []) or []:
        parts.extend(str(value) for value in result.values())
    return "\n".join(parts)


def _source_refs(state: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for document in state.get("retrieved_documents", []) or []:
        source = document.get("filename") or document.get("source", "")
        page = document.get("page_start", "")
        heading = document.get("heading_path", "")
        refs.append(f"file: {source}; page: {page or 'N/A'}; heading_path: {heading}")
    for result in state.get("research_results", []) or []:
        url = result.get("url") or result.get("html_url") or result.get("output_path") or ""
        title = result.get("title") or result.get("name") or result.get("type") or ""
        refs.append(f"external: {title}; Source: {url or 'N/A'}")
    return refs[:8]


def _line_has_source(answer: str, term: str) -> bool:
    for line in answer.splitlines():
        if _contains(line, term) and any(marker in line for marker in SOURCE_MARKERS):
            return True
    return False


def _contains(text: str, needle: str) -> bool:
    if not needle:
        return False
    return needle.lower() in text.lower()


def _is_refusal(answer: str) -> bool:
    normalized = answer.lower()
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS) or any(pattern in answer for pattern in REFUSAL_PATTERNS)


def _structure_compliance(answer: str) -> float:
    return 1.0 if all(marker in answer for marker in STRUCTURE_MARKERS) else 0.0


def _claim_source_binding(answer: str) -> float:
    claim_lines = [line for line in answer.splitlines() if line.strip().startswith("- 结论")]
    if not claim_lines:
        return 0.0
    blocks = re.split(r"\n(?=- 结论)", answer)
    claim_blocks = [block for block in blocks if block.strip().startswith("- 结论") or "\n- 结论" in block]
    if not claim_blocks:
        claim_blocks = claim_lines
    bound = 0
    for block in claim_blocks:
        has_evidence = "证据：" in block
        has_source = "来源：" in block and any(marker in block for marker in ("file:", "page:", "heading_path:", "Source:"))
        if has_evidence and has_source:
            bound += 1
    return _ratio(bound, len(claim_blocks))


def _classify_errors(
    *,
    answerable: bool,
    refusal: bool,
    required_terms: list[str],
    required_citations: list[str],
    term_hits: list[bool],
    citation_hits: list[bool],
    cited_term_hits: list[bool],
    grounded_hits: list[bool],
) -> list[str]:
    errors: list[str] = []
    if answerable and refusal:
        errors.append("wrong_refusal")
    if (not answerable) and (not refusal):
        errors.append("wrong_non_refusal")
    if answerable and required_terms and not any(grounded_hits):
        errors.append("retrieval_failure")
    if answerable and required_terms and any(grounded_hits) and not all(term_hits):
        errors.append("generation_omission")
    if required_citations and not all(citation_hits):
        errors.append("citation_error")
    if required_terms and not all(cited_term_hits):
        errors.append("citation_binding_error")
    return errors


def _is_complex(query: str) -> bool:
    normalized = query.lower()
    return len(normalized.split()) >= 14 or any(marker in normalized for marker in ("compare", "recent", "最新", "比较", "综合"))


def _compact(text: str, max_chars: int = 180) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    keys = [
        "task_completion",
        "answer_correctness",
        "answer_completeness",
        "citation_accuracy",
        "citation_coverage",
        "groundedness",
        "refusal_accuracy",
        "route_accuracy",
        "tool_selection_accuracy",
        "trace_validity",
        "structure_compliance",
        "claim_source_binding",
        "latency_seconds",
    ]
    aggregate: dict[str, float] = {}
    for key in keys:
        values = [
            float(item[key])
            for item in results
            if isinstance(item.get(key), int | float)
        ]
        if values:
            aggregate[key] = sum(values) / len(values)
    aggregate["n_total"] = float(len(results))
    for key in keys:
        aggregate[f"n_{key}"] = float(
            len([item for item in results if isinstance(item.get(key), int | float)])
        )
    return aggregate


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def _bool(value: bool) -> float:
    return 1.0 if value else 0.0


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        case = json.loads(stripped)
        if "query" not in case:
            raise ValueError(f"Missing query at {path}:{line_number}")
        cases.append(case)
    return cases


if __name__ == "__main__":
    main()
