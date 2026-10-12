"""Base interface for research tools."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Minimal interface shared by all research tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> list[dict]:
        """Run the tool and return structured results."""


class ToolTransientError(RuntimeError):
    """A retryable tool failure such as timeout, 500, or SSL interruption."""


class ToolPermanentError(RuntimeError):
    """A non-retryable tool failure such as permission or invalid configuration."""
