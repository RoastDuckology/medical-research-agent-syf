# Architecture

The project is a single-agent LangGraph workflow with specialized nodes, explicit shared state, conditional routing, source-grounded context construction, and post-generation reflection.

## Graph

```text
START
  |
  v
Planner Node
  |-- task_type=daily_brief --> Daily Brief Node --> Reflection Node --> END
  |
  `-- task_type=research ----> Memory Node
                                      |
                                      v
                                Research Node
                                      |
                                      v
                                 Report Node
                                      |
                                      v
                               Reflection Node --> END
```

## State

`ResearchState` carries:

- query, task type, plan, and selected tools;
- retrieved RAG documents and SQLite interaction-memory records;
- external research results and a compact tool trace;
- generated report, provider error, reflection, progress, and context summary.

Nodes receive the current state and return partial updates that LangGraph merges into the shared state.

## Planner and Tools

The planner uses deterministic rules to classify research versus daily-brief tasks and to select arXiv, GitHub, or local-only execution. The research node resolves selected tools through `ToolRegistry`, performs one retry for transient failures, and returns a degraded structured result when recovery fails.

Tool execution supports three modes:

- `live`: real arXiv and GitHub APIs;
- `recorded`: deterministic JSON replay for reproducible evaluation;
- `fault_injection`: timeout, HTTP 500, SSL, rate-limit, and permanent failures.

## Retrieval

Local evidence can come from Markdown, PDF pages, paper notes, and experiment logs. The retrieval pipeline combines:

1. normalized BGE dense embeddings stored in Chroma;
2. an in-process BM25 scorer;
3. weighted score fusion and metadata-aware importance;
4. a Top-20 candidate pool;
5. optional lightweight lexical reranking;
6. optional deterministic multi-query expansion.

PDF metadata preserves file, page range, section title, heading path, and source URLs.

## Context and Generation

`ContextBuilder` converts internal state into typed context packets, applies per-source token budgets, compacts oversized packets, and renders a source-aware prompt. DeepSeek is accessed through an OpenAI-compatible Chat Completions endpoint. Provider failures use bounded retries, model fallback, a cooldown cache, and ultimately a deterministic local fallback report.

The report node appends a machine-checkable source appendix. Reflection checks report availability, source preservation, fallback use, and uncertainty indicators.

## Evaluation

Versioned JSONL datasets and command-line evaluators cover:

- Recall@K, Hit@K, MRR, PDF page hit, and OOD false positives;
- answer correctness, completeness, groundedness, and refusal accuracy;
- citation filename checks, citation coverage, and claim-source format binding;
- routing, tool selection, trace validity, and task completion;
- deterministic tool-failure recovery.

Metric definitions, denominators, current results, and known limitations are documented in `evaluation/CURRENT_RESULTS.md`.
