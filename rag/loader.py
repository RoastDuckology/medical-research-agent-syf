"""Markdown document loading utilities for the local research knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from rag.pdf_loader import load_pdf_documents


logger = logging.getLogger(__name__)


def load_markdown_documents(logs_dir: str) -> list[Document]:
    """Load Markdown files from a directory as LangChain documents."""

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
            content = file_path.read_text(encoding="utf-8").lstrip("\ufeff")
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


def load_research_documents(
    logs_dir: str,
    extra_markdown_files: list[str | Path] | tuple[str | Path, ...] = (),
    paper_notes_dir: str | Path = "data/paper_notes",
    papers_dir: str | Path | None = None,
    paper_text_dir: str | Path | None = None,
    enable_pdf_ingestion: bool = True,
) -> list[Document]:
    """Load daily Markdown logs plus explicitly configured research documents."""

    documents = load_markdown_documents(logs_dir)
    seen_sources = {document.metadata.get("source", "") for document in documents}

    for markdown_file in extra_markdown_files:
        file_path = Path(markdown_file)
        if not file_path.exists() or not file_path.is_file():
            logger.warning("Extra Markdown file does not exist: %s", file_path)
            continue
        resolved_source = str(file_path.resolve())
        if resolved_source in seen_sources:
            continue

        try:
            content = file_path.read_text(encoding="utf-8").lstrip("\ufeff")
        except OSError as exc:
            logger.error("Failed to read extra Markdown file %s: %s", file_path, exc)
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": resolved_source,
                    "filename": file_path.name,
                    "date": "",
                    "document_type": _document_type_for(file_path),
                },
            )
        )
        seen_sources.add(resolved_source)

    notes_directory = Path(paper_notes_dir)
    if notes_directory.exists():
        for note_file in sorted(notes_directory.glob("*.md")):
            resolved_source = str(note_file.resolve())
            if resolved_source in seen_sources:
                continue
            try:
                content = note_file.read_text(encoding="utf-8").lstrip("\ufeff")
            except OSError as exc:
                logger.error("Failed to read paper note %s: %s", note_file, exc)
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": resolved_source,
                        "filename": note_file.name,
                        "date": "",
                        "document_type": "paper_note",
                    },
                )
            )
            seen_sources.add(resolved_source)

    if enable_pdf_ingestion and papers_dir and paper_text_dir:
        for document in load_pdf_documents(papers_dir, paper_text_dir):
            resolved_source = document.metadata.get("source", "")
            page = document.metadata.get("page_start", "")
            dedupe_key = f"{resolved_source}#page={page}"
            if dedupe_key in seen_sources:
                continue
            documents.append(document)
            seen_sources.add(dedupe_key)

    return documents


def _document_type_for(file_path: Path) -> str:
    if file_path.name == "实验进度记录.md":
        return "experiment_progress_log"
    return "research_markdown"
