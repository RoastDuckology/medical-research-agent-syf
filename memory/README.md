# Memory Module

This module implements a small local long-term memory layer inspired by chapter 8.

- Working memory still lives inside `ResearchState` during one LangGraph run.
- Semantic research memory comes from the Markdown/PDF-ready RAG index.
- Episodic memory is stored in SQLite after each completed chat turn.

Default database:

```text
data/memory/agent_memory.sqlite3
```

Set a custom path with:

```text
MEMORY_DB_PATH=data/memory/agent_memory.sqlite3
```
