"""Text splitting utilities for local Markdown documents."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks while preserving metadata."""

    if not documents:
        return []

    settings = get_settings()
    chunks: list[Document] = []
    for document in documents:
        splitter = _splitter_for_document(document, settings.chunk_size, settings.chunk_overlap)
        structured_sections = _split_markdown_sections(document)
        source_documents = structured_sections or [document]
        document_chunks = splitter.split_documents(source_documents)
        for chunk_index, chunk in enumerate(document_chunks):
            chunk.metadata = {
                **document.metadata,
                **chunk.metadata,
                "chunk_index": chunk_index,
                "chunk_text_length": len(chunk.page_content),
                "chunk_preview": _preview(chunk.page_content),
            }
            chunks.append(chunk)

    return chunks


def _splitter_for_document(
    document: Document,
    default_chunk_size: int,
    default_chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """Use different chunk sizes for logs, notes, and page-level PDF text."""

    document_type = str(document.metadata.get("document_type", ""))
    if document_type == "paper_pdf":
        chunk_size = max(default_chunk_size, 900)
        chunk_overlap = max(default_chunk_overlap, 120)
    elif document_type == "paper_note":
        chunk_size = max(default_chunk_size, 650)
        chunk_overlap = max(default_chunk_overlap, 100)
    elif document_type in {"experiment_progress_log", "daily_log"}:
        chunk_size = min(default_chunk_size, 520)
        chunk_overlap = min(default_chunk_overlap, 90)
    else:
        chunk_size = default_chunk_size
        chunk_overlap = default_chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DATE_HEADING_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}")


def _split_markdown_sections(document: Document) -> list[Document]:
    """Split Markdown by headings while retaining date and section metadata."""

    if not _looks_like_markdown(document):
        return []

    lines = document.page_content.splitlines()
    sections: list[Document] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_metadata = dict(document.metadata)
    current_date = str(document.metadata.get("date", ""))

    def flush() -> None:
        if not current_lines:
            return
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines.clear()
            return
        sections.append(Document(page_content=content, metadata=dict(current_metadata)))
        current_lines.clear()

    for line in lines:
        normalized_line = line.lstrip("\ufeff")
        match = HEADING_RE.match(normalized_line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip().lstrip("\ufeff")
            heading_stack[:] = [
                (existing_level, existing_title)
                for existing_level, existing_title in heading_stack
                if existing_level < level
            ]
            heading_stack.append((level, title))
            if level <= 2 and DATE_HEADING_RE.match(title):
                current_date = title[:10]

            current_metadata = {
                **document.metadata,
                "date": current_date,
                "date_heading": current_date,
                "heading_level": level,
                "section_title": title,
                "heading_path": _heading_path(heading_stack, current_date),
            }
        current_lines.append(line)

    flush()
    return sections


def _looks_like_markdown(document: Document) -> bool:
    filename = str(document.metadata.get("filename", ""))
    return filename.endswith(".md") or "\n#" in document.page_content


def _heading_path(heading_stack: list[tuple[int, str]], current_date: str) -> str:
    headings = [heading for _, heading in heading_stack]
    if current_date and (not headings or not headings[0].startswith(current_date)):
        headings.insert(0, current_date)
    return " > ".join(headings)


def _preview(text: str, max_chars: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."
