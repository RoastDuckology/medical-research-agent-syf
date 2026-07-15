# Architecture

This project implements a personal research assistant agent for medical AI and computer vision research.

## Workflow

```text
User Query
  |
  v
LangGraph Runtime
  |
  v
Memory Node
  |
  v
RAG Retriever
  |
  v
Research Node
  |
  v
Tool Registry
  |
  v
External Tools
  |
  v
LLM Report Generation
```

## State

`ResearchState` is the shared data contract between LangGraph nodes. It contains:

- `query`: user question.
- `memory`: formatted personal research memory.
- `retrieved_documents`: structured chunks retrieved from the local knowledge base.
- `research_results`: structured external research results from tools.
- `report`: final generated Markdown report.

## Nodes

`memory_node` retrieves relevant personal research notes from the local vector database.

`research_node` calls registered research tools through the tool registry. It does not perform planning or autonomous tool selection.

`report_node` builds a structured prompt from memory and research results, then calls the configured LLM provider.

## RAG Memory

The RAG pipeline loads Markdown notes, splits them into chunks, embeds them with a local Hugging Face embedding model, and stores them in ChromaDB.

```text
Markdown -> Loader -> Splitter -> Embedding -> Chroma -> Retriever
```

## Tools

Tools follow a small `BaseTool` interface:

```python
run(**kwargs) -> list[dict]
```

The registry keeps the research node independent from concrete tool implementations.

## LLM Layer

The LLM layer uses a common `BaseLLM` interface. If `DEEPSEEK_API_KEY` is configured, the project uses `DeepSeekLLM`. Otherwise, it falls back to `MockLLM` so the demo remains runnable without credentials.

