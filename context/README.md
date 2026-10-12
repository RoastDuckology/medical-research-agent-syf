# Context Engineering

This module implements a lightweight GSSC context pipeline inspired by chapter 9:

```text
Gather -> Select -> Structure -> Compress
```

## Responsibilities

- Gather context from the current task, plan, RAG evidence, episodic memory, tool traces, and external results.
- Select context packets by type-specific relevance and importance under `CONTEXT_BUDGET_TOKENS`.
- Structure the final LLM input with stable sections:
  - `[Role & Policies]`
  - `[Task]`
  - `[Plan]`
  - `[Evidence: Local RAG / PDF]`
  - `[Context: Episodic Memory]`
  - `[Tool Trace]`
  - `[External Results]`
  - `[Output Requirements]`
- Compress oversized packets while preserving source metadata such as file names, pages, heading paths, refs, and URLs.

The public entry point is `ContextBuilder.build(state)`.
