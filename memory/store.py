"""SQLite-backed long-term memory for agent interactions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import get_settings
from memory.scoring import final_memory_score, lexical_overlap_score


@dataclass(frozen=True)
class MemoryRecord:
    """One searchable memory record from a previous agent interaction."""

    id: int
    created_at: str
    memory_type: str
    query: str
    summary: str
    source_refs: list[str]
    importance: float
    memory_score: float


class SQLiteMemoryStore:
    """Small local memory store inspired by chapter 8 MemoryTool."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else get_settings().memory_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_interaction(self, state: dict[str, Any]) -> int:
        """Store the final state from one completed agent turn."""

        query = str(state.get("query", "")).strip()
        if not query:
            raise ValueError("Cannot store memory without a query.")

        report = str(state.get("report", "")).strip()
        summary = _compact_text(report, max_chars=1000)
        memory_type = _memory_type_for(state)
        source_refs = _extract_source_refs(state)
        importance = _estimate_importance(state, source_refs)

        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    created_at, memory_type, query, summary, source_refs,
                    importance, raw_state_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    memory_type,
                    query,
                    summary,
                    json.dumps(source_refs, ensure_ascii=False),
                    importance,
                    json.dumps(_serializable_state(state), ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def search(self, query: str, limit: int = 3) -> list[MemoryRecord]:
        """Search episodic interaction memory with recency and importance scoring."""

        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, memory_type, query, summary, source_refs, importance
                FROM memories
                ORDER BY created_at DESC
                LIMIT 200
                """
            ).fetchall()

        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            text = f"{row['query']} {row['summary']}".lower()
            base_similarity = lexical_overlap_score(normalized_query, text)
            score = final_memory_score(
                base_similarity=base_similarity,
                timestamp=str(row["created_at"]),
                importance=float(row["importance"]),
                half_life_days=90.0,
            )
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    MemoryRecord(
                        id=int(row["id"]),
                        created_at=str(row["created_at"]),
                        memory_type=str(row["memory_type"]),
                        query=str(row["query"]),
                        summary=str(row["summary"]),
                        source_refs=json.loads(row["source_refs"] or "[]"),
                        importance=float(row["importance"]),
                        memory_score=score,
                    ),
                )
            )

        return [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]

    def stats(self) -> dict[str, Any]:
        """Return basic memory store statistics."""

        with sqlite3.connect(self.db_path) as connection:
            total = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_type = connection.execute(
                "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
            ).fetchall()
        return {
            "path": str(self.db_path),
            "total": int(total),
            "by_type": {memory_type: int(count) for memory_type, count in by_type},
        }

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    query TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_refs TEXT NOT NULL,
                    importance REAL NOT NULL,
                    raw_state_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"
            )


def _memory_type_for(state: dict[str, Any]) -> str:
    task_type = state.get("task_type")
    if task_type == "daily_brief":
        return "episodic_daily_brief"
    if state.get("retrieved_documents"):
        return "semantic_research_qa"
    return "episodic_interaction"


def _extract_source_refs(state: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for document in state.get("retrieved_documents", []) or []:
        source = document.get("source") or document.get("filename")
        heading = document.get("heading_path")
        if source and heading:
            refs.append(f"{source}#{heading}")
        elif source:
            refs.append(str(source))
    for result in state.get("research_results", []) or []:
        for key in ("url", "html_url", "output_path"):
            value = result.get(key)
            if value:
                refs.append(str(value))
    return list(dict.fromkeys(refs))


def _estimate_importance(state: dict[str, Any], source_refs: list[str]) -> float:
    score = 0.3
    if source_refs:
        score += 0.3
    if state.get("reflection", {}).get("passed"):
        score += 0.2
    if state.get("task_type") == "research":
        score += 0.1
    if state.get("report_error"):
        score -= 0.2
    return max(0.0, min(1.0, score))


def _serializable_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key
        in {
            "query",
            "task_type",
            "selected_tools",
            "reasoning_trace",
            "retrieved_documents",
            "research_results",
            "report",
            "report_error",
            "reflection",
            "context_summary",
        }
    }


def _compact_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."
