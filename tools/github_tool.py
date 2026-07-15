"""GitHub repository research tool backed by the GitHub REST API."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

from config.settings import get_settings
from tools.base import BaseTool


logger = logging.getLogger(__name__)

GITHUB_SEARCH_REPOSITORIES_URL = "https://api.github.com/search/repositories"


class GithubTool(BaseTool):
    """Search repositories through the GitHub REST API."""

    name = "github"
    description = "Search GitHub repositories for a research query."

    def run(self, **kwargs: Any) -> list[dict]:
        """Return repository search results from GitHub."""

        query = str(kwargs.get("query", "")).strip()
        if not query:
            return []

        settings = get_settings()
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": str(settings.github_max_results),
        }
        url = f"{GITHUB_SEARCH_REPOSITORIES_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers=_github_headers(settings.github_token))

        try:
            with urllib.request.urlopen(request, timeout=settings.request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("GitHub API request failed for query '%s': %s", query, exc)
            return []

        return _parse_github_response(payload)


def _github_headers(token: str) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "medical-research-agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_github_response(payload: dict[str, Any]) -> list[dict]:
    """Parse GitHub repository search JSON into the project result shape."""

    results: list[dict] = []
    for item in payload.get("items", []):
        results.append(
            {
                "type": "repository",
                "source": "github",
                "name": item.get("full_name") or item.get("name", ""),
                "description": item.get("description") or "",
                "stars": int(item.get("stargazers_count") or 0),
                "url": item.get("html_url", ""),
                "language": item.get("language") or "",
                "updated_at": item.get("updated_at", ""),
            }
        )

    return results


def search_github(query: str) -> list[dict]:
    """Backward-compatible helper for direct GitHub repository search."""

    return GithubTool().run(query=query)
