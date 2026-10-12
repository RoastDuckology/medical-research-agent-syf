# Medical AI Research Assistant Agent

A source-grounded research agent for medical AI and computer-vision workflows. The project combines conditional LangGraph routing, hybrid local retrieval, external research tools, bounded-context report generation, reflection, and reproducible evaluation.

This public repository contains sanitized demo data and fixed evaluation cases. It does **not** contain private research notes, API keys, local vector databases, interaction-memory databases, or copyrighted paper PDFs.

## Highlights

- **Conditional agent workflow**: planner, local memory/RAG, external research, report generation, daily brief, and reflection nodes.
- **Hybrid retrieval**: dense embeddings plus BM25, Top-20 candidate pooling, lightweight reranking, and adaptive multi-query expansion.
- **PDF and Markdown ingestion**: page, file, section, and heading metadata are preserved for source-aware answers.
- **Tool orchestration**: arXiv, GitHub, and daily-brief tools behind a common registry.
- **Reliability controls**: bounded retries, provider cooldown, fallback generation, recorded-tool replay, and deterministic fault injection.
- **Evaluation suite**: retrieval, local QA, routing, citation-format, refusal, efficiency, and failure-recovery metrics.

## Architecture

```text
User Query
    |
    v
Planner Node
    |-- daily brief --> Daily Brief Node --> Reflection --> END
    |
    `-- research ----> Memory Node
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

The planner uses deterministic, inspectable routing rules. It selects only the tools needed for a query. This is a single-agent LangGraph workflow with specialized nodes, not a multi-agent system.

## Retrieval Pipeline

```text
Markdown / PDF
    -> metadata-aware chunking
    -> BGE dense embeddings + Chroma
    -> BM25 keyword retrieval
    -> candidate fusion (0.7 dense / 0.3 keyword)
    -> Top-20 candidate pool
    -> lightweight lexical reranker
    -> Top-K evidence
```

Multi-query expansion is implemented as a lightweight deterministic query expander. The adaptive evaluation mode enables expansion only for complex queries.

## Reproducible Evaluation

The repository includes versioned, sanitized evaluation cases and scripts under `data/eval/` and `evaluation/`.

Current fixed retrieval results on 60 medical-AI research cases:

| Mode | Recall@5 | MRR |
|---|---:|---:|
| Dense-only baseline | 55.00% | 0.4644 |
| Dense + BM25 | 61.67% | 0.5200 |
| Hybrid + lightweight rerank | 63.33% | 0.5339 |
| Adaptive MQE + rerank | 63.33% | 0.5339 |

The real-LLM QA run used a fixed `deepseek-chat` model on 60 frozen tasks. Answer correctness and completeness were 94.00% on the 50 answer-labeled tasks; structure compliance was 95.00% across all 60 tasks. These metrics are reported separately from end-to-end task completion and citation accuracy. See [evaluation/CURRENT_RESULTS.md](evaluation/CURRENT_RESULTS.md) for definitions, denominators, limitations, and commands.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

The default embedding model is `BAAI/bge-small-zh-v1.5`. Its first use may require a model download.

## Usage

Build the local index from the included synthetic Markdown logs:

```powershell
python -m rag.indexer
```

Run an interactive CLI:

```powershell
python app.py
```

Run one query:

```powershell
python app.py --query "Summarize the local StereoVLA research notes"
```

Run the Streamlit UI:

```powershell
streamlit run frontend/streamlit_app.py
```

Without `DEEPSEEK_API_KEY`, the project uses `MockLLM` so the workflow remains locally runnable.

## Evaluation Commands

Retrieval ablation:

```powershell
python -m evaluation.evaluate_rag --cases data/eval/rag_eval_cases_v1.jsonl --mode all --top-k 5 --disable-domain-guard --output evaluation/rag_results.local.json
```

Recorded-tool workflow evaluation:

```powershell
python -m evaluation.evaluate_workflow --cases data/eval/agent_routing_cases_v1.jsonl --recorded-tools --output evaluation/workflow_results.local.json
```

Failure-recovery evaluation:

```powershell
python -m evaluation.evaluate_failure_recovery --cases data/eval/failure_recovery_cases_v1.jsonl --output evaluation/failure_recovery.local.json
```

Run tests:

```powershell
pytest -q
```

## Public Data Policy

The repository intentionally excludes:

- `.env` and all credentials;
- Chroma persistence directories;
- SQLite interaction-memory databases;
- private experiment logs and notes;
- paper PDFs and extracted full-text caches;
- generated reports containing private context.

To test PDF ingestion, place documents locally under `data/papers/`; those files remain ignored by Git.

## Known Limitations

- The planner is rule-based and does not cover arbitrary user intent.
- The reranker is a lightweight lexical scorer, not a trained cross-encoder.
- Adaptive MQE matched, but did not exceed, the best hybrid-rerank result on the current retrieval set.
- Citation-format compliance is stronger than semantic citation accuracy; manual citation-support auditing remains incomplete.
- Real-provider evaluations can be affected by timeout, 503, and model-access failures, which are recorded separately rather than silently mixed with fallback outputs.

## License

MIT. See [LICENSE](LICENSE).
