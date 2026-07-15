"""Memory retrieval node backed by the local Markdown vector index."""

from agent.state import ResearchState
from rag.retriever import retrieve_documents


def _format_memory(results: list[dict]) -> str:
    """Convert retrieved chunks into readable memory text."""

    if not results:
        return "未从个人工作日志中检索到相关内容。"

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"{index}. 日期：{result.get('date', '')}",
                    f"   来源文件：{result.get('filename', '')}",
                    f"   正文片段：{result.get('content', '').strip()}",
                ]
            )
        )
    return "\n\n".join(blocks)


def memory_node(state: ResearchState) -> dict:
    """Retrieve personal memory relevant to the user's query."""

    results = retrieve_documents(state["query"])
    return {
        "retrieved_documents": results,
        "memory": _format_memory(results),
    }
