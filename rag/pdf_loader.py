"""PDF extraction and loading for full-paper RAG."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def load_pdf_documents(
    papers_dir: str | Path,
    paper_text_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> list[Document]:
    """Extract PDFs to Markdown cache files and load page-level documents."""

    papers_path = Path(papers_dir)
    text_path = Path(paper_text_dir)
    if not papers_path.exists():
        logger.warning("PDF directory does not exist: %s", papers_path)
        return []

    text_path.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path or papers_path / "manifest.jsonl")
    documents: list[Document] = []

    for pdf_path in sorted(papers_path.glob("*.pdf")):
        metadata = manifest.get(_manifest_key(pdf_path), {})
        cache_path = text_path / f"{pdf_path.stem}.md"
        if _should_refresh_cache(pdf_path, cache_path):
            pages = extract_pdf_pages(pdf_path)
            cache_path.write_text(
                _format_pages_as_markdown(pdf_path, pages, metadata),
                encoding="utf-8",
            )
        else:
            pages = _read_cached_pages(cache_path)

        for page in pages:
            page_text = page["text"].strip()
            if not page_text:
                continue
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": str(pdf_path.resolve()),
                        "filename": pdf_path.name,
                        "document_type": "paper_pdf",
                        "paper_title": metadata.get("title", pdf_path.stem),
                        "arxiv_id": metadata.get("arxiv_id", ""),
                        "abs_url": metadata.get("abs_url", ""),
                        "pdf_url": metadata.get("pdf_url", ""),
                        "page_start": page["page"],
                        "page_end": page["page"],
                        "section_title": f"Page {page['page']}",
                        "heading_path": f"{metadata.get('title', pdf_path.stem)} > Page {page['page']}",
                        "text_cache": str(cache_path.resolve()),
                    },
                )
            )

    return documents


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract page text from a PDF, preferring pdfplumber with pypdf fallback."""

    path = Path(pdf_path)
    pages = _extract_with_pdfplumber(path)
    if not any(page["text"].strip() for page in pages):
        pages = _extract_with_pypdf(path)
    return pages


def _extract_with_pdfplumber(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - covered by dependency checks.
        raise RuntimeError("Install pdfplumber to ingest PDF files.") from exc

    pages: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                pages.append({"page": index, "text": _clean_pdf_text(text)})
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s", pdf_path, exc)
        return []
    return pages


def _extract_with_pypdf(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pypdf to ingest PDF files.") from exc

    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": index, "text": _clean_pdf_text(text)})
    return pages


def _format_pages_as_markdown(
    pdf_path: Path,
    pages: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    title = metadata.get("title", pdf_path.stem)
    lines = [
        f"# {title}",
        "",
        f"- Local PDF: `{pdf_path.as_posix()}`",
        f"- arXiv ID: {metadata.get('arxiv_id', '')}",
        f"- Source: {metadata.get('abs_url', '')}",
        f"- PDF: {metadata.get('pdf_url', '')}",
        "",
    ]
    for page in pages:
        lines.extend(
            [
                f"## Page {page['page']}",
                "",
                page["text"].strip(),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _read_cached_pages(cache_path: Path) -> list[dict[str, Any]]:
    text = cache_path.read_text(encoding="utf-8")
    pages: list[dict[str, Any]] = []
    pattern = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages.append({"page": int(match.group(1)), "text": text[start:end].strip()})
    return pages


def _load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        pdf_file = Path(item.get("pdf_file", "")).name
        if pdf_file:
            records[pdf_file] = item
    return records


def _manifest_key(pdf_path: Path) -> str:
    return pdf_path.name


def _should_refresh_cache(pdf_path: Path, cache_path: Path) -> bool:
    return not cache_path.exists() or cache_path.stat().st_mtime < pdf_path.stat().st_mtime


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
