"""DeepSeek LLM provider.

DeepSeek exposes an OpenAI-compatible chat completions API. This module keeps
the HTTP call isolated so agent nodes do not depend on provider details.
"""

from __future__ import annotations

import requests

from config.settings import get_settings
from llm.base import BaseLLM


class DeepSeekLLM(BaseLLM):
    """Generate text with DeepSeek chat completions."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using DeepSeek."""

        settings = get_settings()
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        endpoint = _chat_completions_endpoint(settings.deepseek_base_url)
        payload = {
            "model": settings.llm_model or "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical AI research assistant. Generate a "
                        "clear, structured Markdown research report in Chinese."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "medical-research-agent/0.1",
                },
                timeout=settings.request_timeout,
            )
            response.raise_for_status()
            response_payload = response.json()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            detail = exc.response.text if exc.response is not None else ""
            raise RuntimeError(f"DeepSeek API request failed: {status_code} {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

        try:
            return response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DeepSeek API response: {response_payload}") from exc


def _chat_completions_endpoint(base_url: str) -> str:
    """Build a chat completions endpoint from official or third-party base URLs."""

    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if "api.deepseek.com" in base:
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"
