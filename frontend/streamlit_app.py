"""Streamlit web demo for the Medical AI Research Assistant Agent."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.workflow import invoke
from rag.indexer import build_or_update_index
from rag.vector_store import vector_store_count


DEFAULT_QUERY = "总结近期 StereoVLA 相关工作"


def _ensure_knowledge_base() -> None:
    """Build the demo knowledge base if the local vector store is empty."""

    if vector_store_count() == 0:
        with st.spinner("Building local demo knowledge base..."):
            stats = build_or_update_index()
        st.info(
            "Knowledge base initialized: "
            f"{stats['markdown_files']} Markdown files, "
            f"{stats['chunks']} chunks, "
            f"{stats['added']} vectors added."
        )


def _run_agent(query: str) -> dict:
    """Invoke the existing LangGraph workflow."""

    initial_state = {
        "query": query,
        "memory": "",
        "retrieved_documents": [],
        "research_results": [],
        "report": "",
    }
    return invoke(initial_state)


def _render_research_results(results: list[dict]) -> None:
    """Render external research results grouped by result type."""

    papers = [item for item in results if item.get("type") == "paper"]
    repositories = [item for item in results if item.get("type") == "repository"]

    st.subheader("arXiv Papers")
    if papers:
        for paper in papers:
            title = paper.get("title", "Untitled paper")
            url = paper.get("url", "")
            st.markdown(f"- [{title}]({url})" if url else f"- {title}")
            abstract = paper.get("abstract", "")
            if abstract:
                st.caption(abstract[:500])
    else:
        st.caption("No paper results returned.")

    st.subheader("GitHub Repositories")
    if repositories:
        for repo in repositories:
            name = repo.get("name", "unknown/repository")
            url = repo.get("url", "")
            stars = repo.get("stars", 0)
            description = repo.get("description", "")
            st.markdown(f"- [{name}]({url}) - {stars} stars" if url else f"- {name} - {stars} stars")
            if description:
                st.caption(description)
    else:
        st.caption("No repository results returned.")


def main() -> None:
    """Render the Streamlit web demo."""

    st.set_page_config(
        page_title="Medical AI Research Assistant Agent",
        page_icon="M",
        layout="wide",
    )

    st.title("Medical AI Research Assistant Agent")
    st.caption("LangGraph + RAG + Tool Calling + DeepSeek-compatible report generation")

    query = st.text_area(
        "Research Question",
        value=DEFAULT_QUERY,
        height=100,
    )

    if st.button("Generate Report", type="primary"):
        normalized_query = query.strip()
        if not normalized_query:
            st.warning("Please enter a research question.")
            return

        try:
            _ensure_knowledge_base()
            with st.spinner("Running Agent workflow..."):
                final_state = _run_agent(normalized_query)
        except Exception as exc:
            st.error(f"Agent execution failed: {exc}")
            return

        st.divider()

        st.header("Query")
        st.write(final_state["query"])

        st.header("Retrieved Memory")
        with st.expander("View retrieved personal research memory", expanded=True):
            st.markdown(final_state.get("memory") or "No memory retrieved.")

        st.header("External Research")
        _render_research_results(final_state.get("research_results", []))

        st.header("Final Report")
        st.markdown(final_state.get("report") or "No report generated.")


if __name__ == "__main__":
    main()
