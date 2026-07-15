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
