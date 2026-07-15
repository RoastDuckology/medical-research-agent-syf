# Architecture

Medical AI Research Assistant Agent is organized as a compact LangGraph workflow. It combines local personal memory retrieval, external research tools, and LLM-based report generation.

## System Flow

```text
User Query
  |
  v
LangGraph Workflow
  |
  v
Memory Node
  |
  v
RAG Retriever
  |
  v
Chroma Vector Database
  |
  v
Research Node
  |
  v
Tool Registry
  |
  v
arXiv API / GitHub API
  |
  v
Report Node
  |
  v
DeepSeek-compatible LLM
  |
  v
Research Report
```

## Components

### State

`ResearchState` is the shared execution state passed between LangGraph nodes.

It contains:

- `query`: user question.
- `memory`: formatted personal research memory.
- `retrieved_documents`: structured RAG results.
- `research_results`: external paper and repository results.
- `report`: final generated Markdown report.

### Node

Nodes are deterministic workflow steps:

- `memory_node`: retrieves relevant personal notes.
- `research_node`: calls registered research tools.
- `report_node`: builds a report prompt and calls the LLM provider.

The current workflow does not include a Planner or dynamic routing.

### Tool

Tools expose external capabilities through a common `BaseTool` interface.

```python
run(**kwargs) -> list[dict]
```

`ToolRegistry` keeps the research node independent from concrete tool implementations.

### RAG

The RAG module builds personal memory from Markdown logs:

```text
Markdown -> Loader -> Splitter -> Embedding -> Chroma -> Retriever
```

This allows the agent to answer based on local research notes rather than only external sources.

### LLM

The LLM layer uses a `BaseLLM` interface. The project supports:

- `DeepSeekLLM` for real report generation.
- `MockLLM` for local demos without credentials.

The report node depends on the interface, not on a specific provider.

