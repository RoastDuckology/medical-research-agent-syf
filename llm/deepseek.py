"""OpenAI-compatible DeepSeek provider with bounded retries."""

from __future__ import annotations

import time

import requests

from config.settings import get_settings
from llm.base import BaseLLM
from llm.health import clear_failure, mark_failure, recent_failure


class DeepSeekLLM(BaseLLM):
    """Generate text with DeepSeek-compatible chat completions."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using configured primary/fallback models."""

        settings = get_settings()
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        endpoint = _chat_completions_endpoint(settings.deepseek_base_url)
        cached_failure = recent_failure(settings.llm_failure_cooldown_seconds)
        if cached_failure:
            raise RuntimeError(f"LLM provider is in cooldown after a recent failure: {cached_failure}")

        errors: list[str] = []
        attempts = max(settings.llm_retry_attempts, 1)
        for model in _candidate_models(settings.llm_model, settings.llm_fallback_models):
            for attempt in range(1, attempts + 1):
                try:
                    result = _request_completion(endpoint, prompt, model)
                    clear_failure()
                    return result
                except RuntimeError as exc:
                    message = str(exc)
                    errors.append(f"{model} attempt {attempt}: {message}")
                    if _is_model_unavailable_error(message):
                        break
                    if attempt < attempts:
                        time.sleep(min(2 ** (attempt - 1), 4))
                        continue
                    final_error = "DeepSeek API request failed after network/provider error: " + message
                    mark_failure(final_error)
                    raise RuntimeError(final_error) from exc

        final_error = "DeepSeek API request failed after retries: " + " | ".join(errors[-4:])
        mark_failure(final_error)
        raise RuntimeError(final_error)


def _request_completion(endpoint: str, prompt: str, model: str) -> str:
    settings = get_settings()
    payload = {
        "model": model or "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical AI research assistant. Generate a concise, "
                    "source-grounded Markdown answer in Chinese. Preserve file names, "
                    "page numbers, heading_path, and Source URLs exactly when provided. "
                    "For every conclusion, include evidence and a source line."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": settings.llm_max_tokens,
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
        raise RuntimeError(f"{status_code} {detail}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        content = str(response_payload["choices"][0]["message"].get("content") or "").strip()
        if not content:
            raise RuntimeError(f"Empty model content: {response_payload}")
        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {response_payload}") from exc


def _candidate_models(primary_model: str, fallback_models: tuple[str, ...]) -> list[str]:
    models: list[str] = []
    for model in (primary_model, *fallback_models):
        if model and model not in models:
            models.append(model)
    return models


def _is_model_unavailable_error(message: str) -> bool:
    lowered = message.lower()
    return "model_not_found" in lowered or "no available channel" in lowered


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
