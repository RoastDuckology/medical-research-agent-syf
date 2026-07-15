"""Application entry point for the public demo."""

from agent.workflow import invoke
from rag.indexer import build_or_update_index
from rag.vector_store import vector_store_count


def main() -> None:
    """Build the local index if needed, then run the LangGraph workflow."""

    if vector_store_count() == 0:
        print("Knowledge base is empty. Building index...")
        stats = build_or_update_index()
        print(
            "Index built: "
            f"{stats['markdown_files']} Markdown files, "
            f"{stats['chunks']} chunks, "
            f"{stats['added']} vectors added."
        )

    initial_state = {
        "query": "Summarize recent StereoVLA-related research progress.",
        "memory": "",
        "retrieved_documents": [],
        "research_results": [],
        "report": "",
    }
    final_state = invoke(initial_state)

    print("\nQuery")
    print(final_state["query"])

    print("\nRetrieved Documents")
    if final_state["retrieved_documents"]:
        for index, document in enumerate(final_state["retrieved_documents"], start=1):
            print(
                f"{index}. {document['filename']} | "
                f"date={document['date']} | distance={document['score']:.4f}"
            )
            print(f"   {document['content'].strip()}")
    else:
        print("No documents retrieved.")

    print("\nResearch Results")
    if final_state["research_results"]:
        for index, result in enumerate(final_state["research_results"], start=1):
            if result.get("type") == "paper":
                print(f"{index}. [Paper] {result['title']} ({result['source']})")
            elif result.get("type") == "repository":
                print(f"{index}. [GitHub] {result['name']} | stars={result['stars']}")
    else:
        print("No research results.")

    print("\nMemory")
    print(final_state["memory"])

    print("\nReport")
    print(final_state["report"])


if __name__ == "__main__":
    main()

