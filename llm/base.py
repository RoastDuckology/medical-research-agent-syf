"""Base abstractions for LLM providers."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Minimal text generation interface shared by LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
