# Evaluation Plan

This folder contains fixed evaluation scripts and datasets for the medical research agent.

## Fixed Datasets

- `data/eval/rag_eval_cases_v1.jsonl`: 60 retrieval cases.
- `data/eval/agent_routing_cases_v1.jsonl`: 30 workflow routing cases.
- `data/eval/end_to_end_tasks_v1.jsonl`: 80 frozen end-to-end research-agent tasks.
- `data/eval/trustworthiness_cases_v1.jsonl`: seed trustworthiness cases for citation and grounding audits.

Keep versioned files immutable after reporting metrics. Add `*_v2.jsonl` for future changes.

## Retrieval Ablation

```powershell
python -m evaluation.evaluate_rag --cases data/eval/rag_eval_cases_v1.jsonl --mode all --top-k 5 --output evaluation/rag_v3_results.json
```

Modes:

| Mode | Dense | BM25 | Candidate Pool | Rerank | Multi-Query |
|---|---:|---:|---:|---:|---:|
| baseline | yes | no | 20 | no | no |
| hybrid | yes | yes | 20 | no | no |
| hybrid_rerank | yes | yes | 20 | yes | no |
| full | yes | yes | 20 | yes | yes |
| adaptive | yes | yes | 20 | yes | query-dependent |

Report:

- `recall@3`, `recall@5`
- `hit@5`
- `mrr`
- `pdf_page_hit`
- `false_positive`
- `full_vs_baseline_recall@5_pp`

Resume-safe wording example:

```text
On 60 fixed medical-AI research retrieval cases, Dense+BM25+MQE improved Recall@5
from X% to Y% over Dense-only, a +Z percentage-point gain.
```

## Agent Workflow

```powershell
python -m evaluation.evaluate_workflow --cases data/eval/agent_routing_cases_v1.jsonl
```

For reproducible external-tool execution:

```powershell
python -m evaluation.evaluate_workflow --cases data/eval/agent_routing_cases_v1.jsonl --recorded-tools --output evaluation/workflow_v4_recorded_results.json
```

Report:

- task routing accuracy
- tool selection accuracy
- tool call success rate
- end-to-end task completion
- fallback success rate

## Trustworthiness

```powershell
python -m evaluation.evaluate_trustworthiness --cases data/eval/trustworthiness_cases_v1.jsonl
```

Report:

- citation coverage
- citation precision
- source location accuracy
- grounded answer rate
- hallucination rate

Citation precision is usually more meaningful than raw citation coverage.

## End-To-End Tasks

```powershell
python -m evaluation.evaluate_end_to_end --cases data/eval/end_to_end_tasks_v1.jsonl --offline --output evaluation/end_to_end_v1_offline.json
```

Report:

- task completion rate
- answer correctness and completeness
- citation accuracy and citation coverage
- groundedness
- refusal accuracy
- trace validity

Use `--offline` for reproducible development numbers. Use the default online mode only when the configured LLM provider and external tools are available, and record provider failures separately.

## Domain Gate

```powershell
python -m evaluation.evaluate_domain_gate --cases data/eval/rag_eval_cases_v1.jsonl --output evaluation/domain_gate_v1.json
```

Report ID and OOD behavior separately:

- ID Recall@5
- ID accept rate
- OOD false accept rate
- refusal accuracy
- coverage

The domain gate should be presented as confidence-based selective refusal, not as a fixed always-on retrieval switch.

## Efficiency

```powershell
python -m evaluation.evaluate_efficiency --cases data/eval/qa_eval_cases.jsonl --output evaluation/efficiency_v2.json
```

Report:

- P50/P95 latency
- average LLM calls
- average tool calls
- average context estimated tokens
- average context packet count
- fallback rate

Only report before/after efficiency gains after running both versions under the same case set.

## Failure Recovery

```powershell
python -m evaluation.evaluate_failure_recovery --cases data/eval/failure_recovery_cases_v1.jsonl --output evaluation/failure_recovery_v1.json
```

Report mock/recorded recovery separately from live network availability:

- retry attempted rate
- recovery policy correctness
- degraded answer success
- fallback success

## Manual Citation Audit

```powershell
python -m evaluation.evaluate_citation_manual_audit --cases data/eval/citation_manual_audit_v1.jsonl --output evaluation/citation_manual_audit_v1.json
```

This is intentionally not a filename-matching metric. Each sampled conclusion
must be manually checked against the cited source, then `source_supports_conclusion`
should be set to `true` or `false`. Unlabeled rows are reported as N/A and are
excluded from the aggregate.
- live tool call success, when real external APIs are enabled
