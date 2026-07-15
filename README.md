# Medical AI Research Assistant Agent

## Overview

Medical AI Research Assistant Agent is a local-first research assistant for medical AI and computer vision researchers.

It combines:

- LangGraph agent workflow
- RAG knowledge retrieval
- Tool calling
- DeepSeek-compatible LLM report generation

The system helps organize personal research notes, retrieve relevant Markdown logs, search external research sources, and generate structured research summaries.

## Features

- LangGraph Workflow
- RAG Knowledge Retrieval
- arXiv Search
- GitHub Repository Search
- LLM-based Research Report Generation
- Local ChromaDB persistence
- Mock LLM fallback for demos without API keys

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full architecture explanation.

```text
User Query
  -> Memory Retrieval
  -> External Research Tools
  -> LLM Report Generation
```

## Demo

Example input:

```text
总结近期 StereoVLA 相关工作
```

The agent retrieves demo Markdown notes from `data/demo_logs/`, searches external research sources, and generates a Markdown report.

See [examples/demo_reports/example_1_stereo_vla_report.md](examples/demo_reports/example_1_stereo_vla_report.md).

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create local environment config:

```bash
copy .env.example .env
```

Optional: configure an LLM key in `.env`.

```text
DEEPSEEK_API_KEY=""
DEEPSEEK_BASE_URL="https://api.deepseek.com"
LLM_MODEL="deepseek-chat"
```

Run the demo:

```bash
python app.py
```

Run tests:

```bash
pytest
```

## Project Structure

```text
agent/      LangGraph state, graph, workflow, and nodes
rag/        Markdown loader, splitter, embeddings, Chroma vector store, retriever
tools/      Tool interface, registry, arXiv tool, GitHub tool
llm/        Base LLM interface, MockLLM, DeepSeek-compatible provider
config/     Environment-based settings
data/       Demo logs and local vector database path
examples/   Example generated reports
docs/       Architecture documentation
```

## Future Work

- Planner node
- Dynamic tool selection
- Better source citation formatting
- Streamlit or FastAPI UI
- PDF ingestion
- More robust evaluation and benchmark scripts

These features are not implemented in the current public demo.

