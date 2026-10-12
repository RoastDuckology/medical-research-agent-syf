"""Build a fixed end-to-end research-agent evaluation set from local annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAG_CASES = Path("data/eval/rag_eval_cases_v1.jsonl")
ROUTING_CASES = Path("data/eval/agent_routing_cases_v1.jsonl")
OUTPUT = Path("data/eval/end_to_end_tasks_v1.jsonl")
LOCAL_OUTPUT = Path("data/eval/local_qa_tasks_v1.jsonl")
WORKFLOW_OUTPUT = Path("data/eval/workflow_tasks_v1.jsonl")


def main() -> None:
    rag_cases = _load_jsonl(RAG_CASES)
    routing_cases = _load_jsonl(ROUTING_CASES)

    local_tasks: list[dict[str, Any]] = []
    for index, case in enumerate(rag_cases[:60], start=1):
        expected_keywords = list(case.get("expected_keywords", []))
        expected_sources = list(case.get("expected_sources", []))
        no_answer = bool(case.get("no_relevant_expected")) or (
            not expected_keywords and not expected_sources and not case.get("expected_pages")
        )
        local_tasks.append(
            {
                "id": f"e2e-local-{index:03d}",
                "task_type": "local_qa",
                "query": case["question"],
                "expected_tool": "local_rag",
                "answerable": not no_answer,
                "relevant_documents": expected_sources,
                "relevant_pages": list(case.get("expected_pages", [])),
                "required_answer_terms": expected_keywords,
                "required_citations": expected_sources,
                "reference_answer": _reference_answer(case, expected_keywords, expected_sources, no_answer),
            }
        )

    workflow_tasks: list[dict[str, Any]] = []
    for index, case in enumerate(routing_cases[:20], start=1):
        expected_tools = list(case.get("expected_tools", []))
        expected_tool = "+".join(expected_tools) if expected_tools else "local_rag"
        workflow_tasks.append(
            {
                "id": f"e2e-workflow-{index:03d}",
                "task_type": "workflow",
                "expected_agent_task_type": case.get("expected_task_type", "research"),
                "query": case["query"],
                "expected_tool": expected_tool,
                "answerable": True,
                "relevant_documents": [],
                "relevant_pages": [],
                "required_answer_terms": _workflow_terms(case),
                "required_citations": [],
                "reference_answer": "The agent should choose the expected workflow and produce a source-aware answer or tool result.",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tasks = local_tasks + workflow_tasks
    OUTPUT.write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + "\n",
        encoding="utf-8",
    )
    LOCAL_OUTPUT.write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in local_tasks) + "\n",
        encoding="utf-8",
    )
    WORKFLOW_OUTPUT.write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in workflow_tasks) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(tasks)} end-to-end tasks to {OUTPUT}")
    print(f"Wrote {len(local_tasks)} local QA tasks to {LOCAL_OUTPUT}")
    print(f"Wrote {len(workflow_tasks)} workflow tasks to {WORKFLOW_OUTPUT}")


def _reference_answer(
    case: dict[str, Any],
    expected_keywords: list[str],
    expected_sources: list[str],
    no_answer: bool,
) -> str:
    if no_answer:
        return "The agent should refuse or state that current local evidence is insufficient."
    keywords = ", ".join(expected_keywords) or "the requested topic"
    sources = ", ".join(expected_sources) or "the retrieved source"
    return f"The answer should mention {keywords} and cite {sources}."


def _workflow_terms(case: dict[str, Any]) -> list[str]:
    expected_tools = list(case.get("expected_tools", []))
    terms: list[str] = []
    if "arxiv" in expected_tools:
        terms.append("arxiv")
    if "github" in expected_tools:
        terms.append("github")
    if "daily_brief" in expected_tools:
        terms.append("brief")
    return terms


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    main()
