"""Tests for LLM providers."""

import pytest

from llm.deepseek import DeepSeekLLM
from llm.mock_llm import MockLLM
from llm.provider import get_llm


def test_mock_llm_generates_local_report():
    report = MockLLM().generate("test prompt")

    assert "MockLLM" in report
    assert "LLM Research Report" in report


def test_provider_uses_mock_without_deepseek_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")

    assert isinstance(get_llm(), MockLLM)


def test_deepseek_llm_parses_chat_completion_response(monkeypatch):
    class FakeResponse:
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "# Report"}}]}

    def fake_post(url, json, headers, timeout):
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_FAILURE_COOLDOWN_SECONDS", "0")
    monkeypatch.setattr("requests.post", fake_post)

    assert DeepSeekLLM().generate("hello") == "# Report"


def test_deepseek_llm_requires_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")

    with pytest.raises(RuntimeError):
        DeepSeekLLM().generate("hello")
