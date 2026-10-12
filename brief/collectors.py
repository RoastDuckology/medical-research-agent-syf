"""External source collectors for the daily AI brief agent."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Any

from config.settings import get_settings
from tools.github_tool import _github_headers


logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
GITHUB_SEARCH_REPOSITORIES_URL = "https://api.github.com/search/repositories"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def collect_recent_arxiv_papers(
    topics: list[str],
    days: int,
    max_results_per_topic: int,
) -> list[dict[str, Any]]:
    """Collect recent arXiv papers for the configured topics."""

    start_date, end_date = _date_window(days)
    seen_urls: set[str] = set()
    results: list[dict[str, Any]] = []

    for topic in topics:
        search_query = _build_arxiv_query(topic, start_date, end_date)
        params = {
            "search_query": search_query,
            "start": "0",
            "max_results": str(max_results_per_topic),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=get_settings().request_timeout) as response:
                payload = response.read()
        except Exception as exc:
            logger.warning("arXiv daily brief request failed for '%s': %s", topic, exc)
            continue

        for item in _parse_arxiv_response(payload, topic):
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            results.append(item)

    return results

def collect_recent_github_repositories(
    topics: list[str],
    days: int,
    max_results_per_topic: int,
) -> list[dict[str, Any]]:
    """Collect recently updated GitHub repositories for the configured topics."""

    start_date, _ = _date_window(days)
    settings = get_settings()
    seen_urls: set[str] = set()
    results: list[dict[str, Any]] = []

    for topic in topics:
        query = f'{topic} pushed:>={start_date.isoformat()}'
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": str(max_results_per_topic),
        }
        url = f"{GITHUB_SEARCH_REPOSITORIES_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers=_github_headers(settings.github_token))

        try:
            with urllib.request.urlopen(request, timeout=settings.request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("GitHub daily brief request failed for '%s': %s", topic, exc)
            continue

        for item in payload.get("items", []):
            html_url = item.get("html_url", "")
            if not html_url or html_url in seen_urls:
                continue
            seen_urls.add(html_url)
            results.append(
                {
                    "type": "repository",
                    "source": "github",
                    "topic": topic,
                    "title": item.get("full_name") or item.get("name", ""),
                    "summary": item.get("description") or "",
                    "url": html_url,
                    "stars": int(item.get("stargazers_count") or 0),
                    "language": item.get("language") or "",
                    "updated_at": item.get("updated_at", ""),
                }
            )

    return results


def _date_window(days: int) -> tuple[date, date]:
    end_date = date.today()
    start_date = end_date - timedelta(days=max(days, 1))
    return start_date, end_date


def _build_arxiv_query(topic: str, start_date: date, end_date: date) -> str:
    start = start_date.strftime("%Y%m%d") + "0000"
    end = end_date.strftime("%Y%m%d") + "2359"
    return f'all:"{topic}" AND submittedDate:[{start} TO {end}]'


def _text(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else " ".join(element.text.split())


def _parse_arxiv_response(payload: bytes, topic: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        logger.warning("Failed to parse arXiv daily brief response: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _text(entry.find("atom:title", ATOM_NS))
        summary = _text(entry.find("atom:summary", ATOM_NS))
        url = _text(entry.find("atom:id", ATOM_NS))
        published = _text(entry.find("atom:published", ATOM_NS))
        authors = [
            _text(author.find("atom:name", ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        venue_hint = _text(entry.find("atom:journal_ref", ATOM_NS))

        results.append(
            {
                "type": "paper",
                "source": "arxiv",
                "topic": topic,
                "title": title,
                "summary": summary,
                "url": url,
                "authors": [author for author in authors if author],
                "published": published,
                "venue_hint": venue_hint,
            }
        )

    return results
