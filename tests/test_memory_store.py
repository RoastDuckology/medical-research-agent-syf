"""Tests for SQLite interaction memory."""

from memory.store import SQLiteMemoryStore


def test_memory_store_add_search_and_stats(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(db_path)

    memory_id = store.add_interaction(
        {
            "query": "StereoVLA baseline 是否可靠",
            "task_type": "research",
            "retrieved_documents": [
                {
                    "source": "实验进度记录.md",
                    "heading_path": "当前判断",
                }
            ],
            "research_results": [],
            "report": "需要进一步验证。file: 实验进度记录.md",
            "reflection": {"passed": True},
        }
    )

    results = store.search("StereoVLA baseline", limit=1)
    stats = store.stats()

    assert memory_id == 1
    assert results
    assert results[0].memory_type == "semantic_research_qa"
    assert results[0].memory_score > 0
    assert stats["total"] == 1
