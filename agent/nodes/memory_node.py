"""Memory retrieval node backed by RAG plus local interaction memory."""

from __future__ import annotations

from agent.state import ResearchState
from memory.store import SQLiteMemoryStore
from rag.retriever import retrieve_documents


def _format_memory(results: list[dict], memory_records: list[dict]) -> str:
    """Convert retrieved evidence into readable, source-grounded memory text."""

    blocks: list[str] = []
    if memory_records:
        blocks.append("## 历史交互记忆")
        for index, record in enumerate(memory_records, start=1):
            refs = "; ".join(record.get("source_refs", []))
            blocks.append(
                "\n".join(
                    [
                        f"{index}. type: {record.get('memory_type', '')}",
                        f"   query: {record.get('query', '')}",
                        f"   summary: {record.get('summary', '')}",
                        f"   refs: {refs}",
                        f"   memory_score: {record.get('memory_score', '')}",
                        f"   importance: {record.get('importance', '')}",
                    ]
                )
            )

    if results:
        blocks.append("## RAG 检索记忆")
        for index, result in enumerate(results, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"{index}. date: {result.get('date', '')}",
                        f"   file: {result.get('filename', '')}",
                        f"   document_type: {result.get('document_type', '')}",
                        f"   heading_path: {result.get('heading_path', '')}",
                        f"   memory_score: {result.get('memory_score', result.get('score', ''))}",
                        f"   dense_score: {result.get('dense_score', '')}",
                        f"   keyword_score: {result.get('keyword_score', '')}",
                        f"   content: {result.get('content', '').strip()}",
                    ]
                )
            )

    if not blocks:
        return "No relevant personal research memory was retrieved."
    return "\n\n".join(blocks)


def memory_node(state: ResearchState) -> dict:
    """Retrieve personal memory relevant to the user's query."""

    query = state["query"]
    results = retrieve_documents(query)
    memory_records = [
        {
            "id": record.id,
            "created_at": record.created_at,
            "memory_type": record.memory_type,
            "query": record.query,
            "summary": record.summary,
            "source_refs": record.source_refs,
            "importance": record.importance,
            "memory_score": record.memory_score,
        }
        for record in SQLiteMemoryStore().search(query)
    ]
    return {
        "retrieved_documents": results,
        "memory_records": memory_records,
        "memory": _format_memory(results, memory_records),
    }
