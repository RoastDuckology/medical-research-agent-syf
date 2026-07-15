"""LLM provider factory."""

from config.settings import get_settings
from llm.base import BaseLLM
from llm.deepseek import DeepSeekLLM
from llm.mock_llm import MockLLM


def get_llm() -> BaseLLM:
    """Return the configured LLM provider.

    DeepSeek is used only when an API key is configured. Otherwise the local
    MockLLM keeps the workflow runnable without external LLM credentials.
    """

    settings = get_settings()
    if settings.deepseek_api_key:
        return DeepSeekLLM()
    return MockLLM()
