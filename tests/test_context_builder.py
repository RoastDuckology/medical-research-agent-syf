"""Tests for GSSC-style context building."""

from context.builder import ContextBuilder


def test_context_builder_structures_and_budgets_context():
    state = {
        "query": "总结 StereoVLA 当前风险",
        "plan": [{"step": "retrieve_memory", "goal": "Find evidence."}],
        "memory_records": [
            {
                "id": 1,
                "memory_type": "episodic_interaction",
                "query": "StereoVLA baseline",
                "summary": "previous answer " * 120,
                "source_refs": ["实验进度记录.md#当前判断"],
                "importance": 0.8,
                "memory_score": 0.9,
            }
        ],
        "retrieved_documents": [
            {
                "filename": "实验进度记录.md",
                "document_type": "experiment_progress_log",
                "heading_path": "2026-06-08 > 目标",
                "content": "StereoVLA evidence " * 200,
                "memory_score": 0.95,
                "importance": 0.9,
            }
        ],
        "reasoning_trace": [{"action": "local_rag_only", "observation": "skip external"}],
        "research_results": [{"type": "paper", "url": "https://arxiv.org/abs/example"}],
    }

    built = ContextBuilder(budget_tokens=900).build(state)

    assert "[Role & Policies]" in built.prompt
    assert "[Evidence: Local RAG / PDF]" in built.prompt
    assert "heading_path:" in built.prompt
    assert built.estimated_tokens <= built.budget_tokens
    assert built.summary()["packet_count"] > 0


def test_context_builder_keeps_task_for_empty_state():
    built = ContextBuilder(budget_tokens=900).build({"query": "hello"})

    assert "[Task]" in built.prompt
    assert "hello" in built.prompt
