"""Markdown document loading utilities for the local research knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def load_markdown_documents(logs_dir: str) -> list[Document]:
    """Load Markdown files from a directory as LangChain documents.

    Each Markdown file becomes one document. Files that fail to read are logged
    and skipped so one broken log does not stop the whole indexing process.
    """

    directory = Path(logs_dir)
    if not directory.exists():
        logger.warning("Markdown logs directory does not exist: %s", directory)
        return []

    markdown_files = sorted(directory.glob("*.md"))
    if not markdown_files:
        logger.warning("No Markdown files found in logs directory: %s", directory)
        return []

    documents: list[Document] = []
    for file_path in markdown_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read Markdown file %s: %s", file_path, exc)
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path.resolve()),
                    "filename": file_path.name,
                    "date": file_path.stem,
                    "document_type": "daily_log",
                },
            )
        )

    return documents
