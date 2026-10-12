"""Tests for interactive chat service helpers."""

from agent.chat_service import ask_agent, format_chat_answer


def test_format_chat_answer_includes_report_reflection_and_trace():
    answer = format_chat_answer(
        {
            "report": "# Report\n\nDone.",
            "reflection": {"passed": False, "issues": ["missing source"]},
            "memory_records": [{"id": 1}],
            "reasoning_trace": [
                {"action": "arxiv", "observation": "1 results returned."}
            ],
        }
    )

    assert "# Report" in answer
    assert "质量检查" in answer
    assert "missing source" in answer
    assert "历史记忆" in answer
    assert "arxiv" in answer


def test_ask_agent_rejects_empty_query():
    try:
        ask_agent("  ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected empty query to raise ValueError")
