"""Tiny on-disk circuit breaker for unstable LLM providers."""

from __future__ import annotations

import json
import time
from pathlib import Path


HEALTH_PATH = Path("data/memory/llm_provider_health.json")


def recent_failure(cooldown_seconds: int) -> str:
    """Return recent failure reason if the provider is still in cooldown."""

    if cooldown_seconds <= 0 or not HEALTH_PATH.exists():
        return ""
    try:
        payload = json.loads(HEALTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    failed_at = float(payload.get("failed_at", 0.0))
    if time.time() - failed_at <= cooldown_seconds:
        return str(payload.get("reason", "recent provider failure"))
    return ""


def mark_failure(reason: str) -> None:
    """Record a provider failure so UI calls can fail fast for a short window."""

    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_PATH.write_text(
        json.dumps(
            {
                "failed_at": time.time(),
                "reason": reason[:1000],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def clear_failure() -> None:
    """Clear provider failure state after a successful call."""

    if HEALTH_PATH.exists():
        HEALTH_PATH.unlink()
