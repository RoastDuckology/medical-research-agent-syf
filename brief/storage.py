"""Storage helpers for generated daily briefs."""

from __future__ import annotations

from pathlib import Path


def save_daily_brief(content: str, brief_date: str, output_dir: str = "data/reports/daily") -> str:
    """Save a daily brief as Markdown and return the output path."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{brief_date}.md"
    output_path.write_text(content, encoding="utf-8")
    return str(output_path)
