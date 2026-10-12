"""Build or update the local Markdown knowledge base index."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings
from rag.loader import load_research_documents
from rag.splitter import split_documents
from rag.vector_store import index_documents


logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def build_or_update_index() -> dict[str, Any]:
    """Load Markdown logs, split them, and index them into Chroma."""

    settings = get_settings()
    documents = load_research_documents(
        str(settings.logs_dir),
        settings.extra_markdown_files,
        settings.paper_notes_dir,
        settings.papers_dir,
        settings.paper_text_dir,
        settings.enable_pdf_ingestion,
    )
    chunks = split_documents(documents)
    index_stats = index_documents(chunks) if chunks else {
        "added": 0,
        "skipped": 0,
        "updated": 0,
        "persist_directory": str(settings.vector_db_path),
    }

    return {
        "markdown_files": len(documents),
        "chunks": len(chunks),
        **index_stats,
    }


def main() -> None:
    """CLI entry point for `python -m rag.indexer`."""

    stats = build_or_update_index()
    print("Knowledge base indexing complete.")
    print(f"Markdown files found: {stats['markdown_files']}")
    print(f"Chunks generated: {stats['chunks']}")
    print(f"Vectors added: {stats['added']}")
    print(f"Duplicates skipped: {stats['skipped']}")
    print(f"Stale vectors updated: {stats['updated']}")
    print(f"Chroma persist directory: {stats['persist_directory']}")


if __name__ == "__main__":
    main()
