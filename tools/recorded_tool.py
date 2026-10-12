"""Recorded research tools for reproducible workflow evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.base import BaseTool


class RecordedTool(BaseTool):
    """Return fixed tool responses from JSON files instead of live network APIs."""

    description = "Replay recorded research tool responses."

    def __init__(self, name: str, recordings_dir: str | Path = "data/recordings") -> None:
        self.name = name
        self._recordings_dir = Path(recordings_dir)

    def run(self, **kwargs: Any) -> list[dict]:
        query = str(kwargs.get("query", "")).lower()
        records = _load_records(self._recordings_dir / f"{self.name}.json")
        if not records:
            return []

        matched = [
            record
            for record in records
            if any(keyword.lower() in query for keyword in record.get("keywords", []))
        ]
        selected = matched or records[:1]
        results: list[dict] = []
        for record in selected:
            for item in record.get("results", []):
                copied = dict(item)
                copied["recorded"] = True
                copied["source"] = copied.get("source", self.name)
                results.append(copied)
        return results


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
