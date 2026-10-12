"""Aggregate human citation-support audit labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CASES = Path("data/eval/citation_manual_audit_v1.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate manual citation support labels.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output", default="", help="Optional UTF-8 JSON output path.")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    labeled = [case for case in cases if isinstance(case.get("source_supports_conclusion"), bool)]
    supported = [case for case in labeled if case["source_supports_conclusion"]]
    payload = json.dumps(
        {
            "case_count": len(cases),
            "labeled_count": len(labeled),
            "unlabeled_count": len(cases) - len(labeled),
            "citation_support_accuracy": (len(supported) / len(labeled)) if labeled else None,
            "cases": cases,
        },
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        case = json.loads(stripped)
        for key in ("conclusion", "citation", "source_supports_conclusion"):
            if key not in case:
                raise ValueError(f"Missing {key} at {path}:{line_number}")
        cases.append(case)
    return cases


if __name__ == "__main__":
    main()
