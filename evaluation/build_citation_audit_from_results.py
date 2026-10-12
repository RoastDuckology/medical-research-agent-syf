"""Build manual citation-support audit samples from generated QA results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path("evaluation/local_qa_v3_llm.json")
DEFAULT_OUTPUT = Path("data/eval/citation_manual_audit_from_llm_v1.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample conclusion-source pairs for manual citation audit.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Result JSON from evaluate_end_to_end.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL audit file.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum audit rows.")
    args = parser.parse_args()

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for case in payload.get("cases", []):
        for conclusion, evidence, source in _extract_claim_blocks(str(case.get("answer", ""))):
            rows.append(
                {
                    "id": f"audit-{len(rows)+1:03d}",
                    "case_id": case.get("id", ""),
                    "query": case.get("query", ""),
                    "conclusion": conclusion,
                    "evidence": evidence,
                    "citation": source,
                    "retrieved_source_refs": case.get("retrieved_source_refs", []),
                    "source_supports_conclusion": None,
                    "error_types": case.get("error_types", []),
                    "notes": "Manually open the cited source/page and set source_supports_conclusion to true or false.",
                }
            )
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} audit rows to {output_path}")


def _extract_claim_blocks(answer: str) -> list[tuple[str, str, str]]:
    blocks = re.split(r"\n(?=- 结论)", answer)
    rows: list[tuple[str, str, str]] = []
    for block in blocks:
        if "结论" not in block or "来源" not in block:
            continue
        conclusion = _field(block, r"结论\d*[:：]\s*(.+)")
        evidence = _field(block, r"证据[:：]\s*(.+)")
        source = _field(block, r"来源[:：]\s*(.+)")
        if conclusion and source:
            rows.append((conclusion, evidence, source))
    return rows


def _field(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if not match:
        return ""
    return match.group(1).strip()


if __name__ == "__main__":
    main()
