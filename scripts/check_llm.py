"""Check configured LLM connectivity without printing secrets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from llm.deepseek import _candidate_models, _chat_completions_endpoint


def main() -> None:
    settings = get_settings()
    endpoint = _chat_completions_endpoint(settings.deepseek_base_url)
    result = {
        "base_url": settings.deepseek_base_url,
        "endpoint": endpoint,
        "has_key": bool(settings.deepseek_api_key),
        "primary_model": settings.llm_model,
        "fallback_models": list(settings.llm_fallback_models),
        "models": [],
    }

    for model in _candidate_models(settings.llm_model, settings.llm_fallback_models):
        result["models"].append(_check_model(endpoint, model))

    print(json.dumps(result, ensure_ascii=False, indent=2))


def _check_model(endpoint: str, model: str) -> dict:
    settings = get_settings()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "只回答 OK"}],
        "temperature": 0.0,
        "max_tokens": 20,
        "stream": False,
    }
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            timeout=min(settings.request_timeout, 20),
        )
        text = response.text[:300].replace("\n", " ")
        return {"model": model, "status": response.status_code, "ok": response.ok, "body": text}
    except Exception as exc:
        return {"model": model, "status": "error", "ok": False, "body": str(exc)}


if __name__ == "__main__":
    main()
