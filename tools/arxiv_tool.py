"""arXiv research tool backed by the public arXiv API."""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from config.settings import get_settings
from tools.base import BaseTool


logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivTool(BaseTool):
    """Search papers through the public arXiv API."""

    name = "arxiv"
    description = "Search arXiv papers for a research query."

    def run(self, **kwargs: Any) -> list[dict]:
        """Return paper search results from arXiv."""

        query = str(kwargs.get("query", "")).strip()
        if not query:
            return []

        settings = get_settings()
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(settings.arxiv_max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=settings.request_timeout) as response:
                payload = response.read()
        except Exception as exc:
            logger.warning("arXiv API request failed for query '%s': %s", query, exc)
            return []

        return _parse_arxiv_response(payload)


def _text(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else " ".join(element.text.split())


def _parse_arxiv_response(payload: bytes) -> list[dict]:
    """Parse arXiv Atom XML into the project's research result shape."""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        logger.warning("Failed to parse arXiv API response: %s", exc)
        return []

    results: list[dict] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _text(entry.find("atom:title", ATOM_NS))
        abstract = _text(entry.find("atom:summary", ATOM_NS))
        url = _text(entry.find("atom:id", ATOM_NS))
        published = _text(entry.find("atom:published", ATOM_NS))
        authors = [
            _text(author.find("atom:name", ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]

        results.append(
            {
                "type": "paper",
                "source": "arxiv",
                "title": title,
                "abstract": abstract,
                "url": url,
                "authors": [author for author in authors if author],
                "published": published,
            }
        )

    return results


def search_arxiv(query: str) -> list[dict]:
    """Backward-compatible helper for direct arXiv search."""

    return ArxivTool().run(query=query)
