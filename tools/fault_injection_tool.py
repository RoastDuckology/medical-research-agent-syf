"""Fault-injection wrappers for tool-node recovery evaluation."""

from __future__ import annotations

from typing import Any

from tools.base import BaseTool, ToolPermanentError, ToolTransientError


class FaultInjectionTool(BaseTool):
    """Raise a configured tool failure when invoked."""

    description = "Inject deterministic tool failures for evaluation."

    def __init__(self, name: str, failure_type: str) -> None:
        self.name = name
        self.failure_type = failure_type

    def run(self, **kwargs: Any) -> list[dict]:
        if self.failure_type in {"timeout", "http_500", "ssl_error", "rate_limit"}:
            raise ToolTransientError(f"{self.name} injected {self.failure_type}")
        raise ToolPermanentError(f"{self.name} injected {self.failure_type}")
