# Current Evaluation Results

Generated on 2026-08-04 from fixed local evaluation files.

## Important Metric Rules

- Answer correctness and answer completeness are computed from the final answer only.
- Retrieved evidence is used for groundedness checks, not for answer correctness.
- Metrics with no keyword/citation labels are `N/A` and excluded from aggregation.
- The 60 local QA tasks and 20 workflow tasks are reported separately.
- Workflow tasks evaluate routing, tool selection, trace validity, and task completion; answer correctness is `N/A`.
- Online LLM metrics are reported separately with fixed `deepseek-chat`; provider
  timeout/503 cases are tracked and rerun rather than silently mixed with fallback.

## Retrieval

Command:

```powershell
python -m evaluation.evaluate_rag --cases data/eval/rag_eval_cases_v1.jsonl --mode all --top-k 5 --disable-domain-guard --output evaluation/rag_v3_no_guard_results.json
```

On 60 fixed medical-AI research retrieval cases:

| Mode | Recall@5 | Hit@5 | MRR | OOD False Positive |
|---|---:|---:|---:|---:|
| Dense-only baseline | 55.00% | 55.00% | 0.4644 | 100.00% |
| Dense+BM25 hybrid | 61.67% | 61.67% | 0.5200 | 100.00% |
| Hybrid + rerank | 63.33% | 63.33% | 0.5339 | 100.00% |
| Full MQE + rerank | 61.67% | 61.67% | 0.5200 | 100.00% |
| Adaptive MQE + rerank | 63.33% | 63.33% | 0.5339 | 100.00% |

Resume-safe wording:

```text
On 60 fixed medical-AI research retrieval cases, adaptive Dense+BM25 retrieval
with Top-20 candidate pooling and lightweight reranking improved Recall@5 from
55.00% to 63.33% over Dense-only, a +8.33 percentage-point gain; MRR improved
from 0.4644 to 0.5339.
```

## Local QA

Command:

```powershell
python -m evaluation.evaluate_end_to_end --cases data/eval/local_qa_tasks_v1.jsonl --offline --task-scope local_qa --output evaluation/local_qa_v2_offline.json
```

On 60 frozen local QA tasks:

| Metric | Result | Aggregated N |
|---|---:|---:|
| Task completion | 83.33% | 60 |
| Answer correctness | 47.00% | 50 |
| Answer completeness | 47.00% | 50 |
| Citation accuracy | 38.00% | 50 |
| Citation coverage | 47.00% | 50 |
| Groundedness | 70.00% | 50 |
| Refusal accuracy | 83.33% | 60 |
| Route accuracy | 95.00% | 60 |
| Tool selection accuracy | 100.00% | 60 |
| Trace validity | 100.00% | 60 |

These are strict offline development metrics. They should not be mixed with
workflow routing metrics.

## Local QA With Real LLM

Provider check:

```powershell
python scripts/check_llm.py
```

The official DeepSeek endpoint is reachable. The stable evaluation run fixes the
model to `deepseek-chat`. A first 60-case pass produced 10 provider errors
because of timeout/503 busy responses; those 10 cases were rerun with a longer
timeout and 3 retries, then merged only when the rerun produced a real LLM answer.

Commands:

```powershell
python -m evaluation.evaluate_end_to_end --cases data/eval/local_qa_tasks_v1.jsonl --task-scope local_qa --output evaluation/local_qa_v4_llm.json
python -m evaluation.evaluate_end_to_end --cases data/eval/local_qa_failed_v4.jsonl --task-scope local_qa --output evaluation/local_qa_v4_failed_rerun.json
```

Merged result:

```text
evaluation/local_qa_v5_llm_merged.json
```

On 60 frozen local QA tasks with fixed `deepseek-chat`:

| Metric | Result | Aggregated N |
|---|---:|---:|
| Task completion | 68.33% | 60 |
| Answer correctness | 94.00% | 50 |
| Answer completeness | 94.00% | 50 |
| Automatic citation filename accuracy | 42.00% | 50 |
| Citation coverage | 53.00% | 50 |
| Groundedness | 67.00% | 50 |
| Refusal accuracy | 68.33% | 60 |
| Structure compliance | 95.00% | 60 |
| Claim-source binding format | 91.67% | 60 |
| Route accuracy | 95.00% | 60 |
| Tool selection accuracy | 93.33% | 60 |
| Trace validity | 93.33% | 60 |

Error type counts:

| Error Type | Count |
|---|---:|
| citation_error | 31 |
| citation_binding_error | 26 |
| wrong_refusal | 17 |
| retrieval_failure | 15 |
| wrong_non_refusal | 2 |
| generation_omission | 1 |

Interpretation:

- The real LLM can follow the structured conclusion/evidence/source format well.
- Current bottlenecks are citation exactness, retrieval coverage for some local
  project facts, and selective refusal calibration.
- `Automatic citation filename accuracy` is only a machine check that the final
  answer mentions the required source string. It is not the same as human citation
  support accuracy.

## Workflow Tasks

Command:

```powershell
python -m evaluation.evaluate_end_to_end --cases data/eval/workflow_tasks_v1.jsonl --offline --task-scope workflow --output evaluation/workflow_tasks_v1_offline.json
```

On 20 frozen workflow tasks:

| Metric | Result |
|---|---:|
| Task completion | 80.00% |
| Refusal accuracy | 80.00% |
| Route accuracy | 100.00% |
| Tool selection accuracy | 100.00% |
| Trace validity | 100.00% |
| Answer correctness/completeness/citation/groundedness | N/A |

Recorded external-tool command:

```powershell
python -m evaluation.evaluate_workflow --cases data/eval/agent_routing_cases_v1.jsonl --recorded-tools --output evaluation/workflow_v4_recorded_results.json
```

On 30 frozen workflow routing cases with recorded arXiv/GitHub/DailyBrief
responses, task routing accuracy, tool selection accuracy, tool call success,
and task completion are all 100.00%.

## Selective Domain Gate

Command:

```powershell
python -m evaluation.evaluate_domain_gate --cases data/eval/rag_eval_cases_v1.jsonl --output evaluation/domain_gate_v1.json
```

Recommended threshold from the current curve:

| Threshold | Coverage | ID Recall@5 | ID Accept Rate | OOD False Accept Rate | Refusal Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 48.33% | 46.00% | 52.00% | 30.00% | 70.00% |

Interpretation: the current gate is useful as a conservative selective-refusal
mode, but should not be the default retrieval switch yet because coverage and ID
recall are still low.

## Failure Recovery

Command:

```powershell
python -m evaluation.evaluate_failure_recovery --cases data/eval/failure_recovery_cases_v1.jsonl --output evaluation/failure_recovery_v2_injected.json
```

On 10 deterministic fault-injection cases that pass through `research_node` for
arXiv/GitHub failures:

| Metric | Result |
|---|---:|
| Retry attempted rate | 60.00% |
| Recovery policy correctness | 100.00% |
| Degraded answer success | 100.00% |
| Fallback success | 100.00% |

## Manual Citation Audit

Command:

```powershell
python -m evaluation.evaluate_citation_manual_audit --cases data/eval/citation_manual_audit_v1.jsonl --output evaluation/citation_manual_audit_v1.json
```

Current status: 5 samples, 0 manually labeled, citation support accuracy is
`N/A`. Fill `source_supports_conclusion=true/false` after opening the cited
source and checking whether it supports the conclusion.

Real LLM audit sample:

```powershell
python -m evaluation.build_citation_audit_from_results --results evaluation/local_qa_v5_llm_merged.json --output data/eval/citation_manual_audit_from_llm_v3.jsonl --limit 30
python -m evaluation.evaluate_citation_manual_audit --cases data/eval/citation_manual_audit_from_llm_v3.jsonl --output evaluation/citation_manual_audit_from_llm_v3.json
```

Current status: 30 sampled conclusion-source pairs, 0 manually labeled,
`citation_support_accuracy=N/A`.

## LLM Provider Status

Command:

```powershell
python scripts/check_llm.py
```

Current provider status:

- Official endpoint `https://api.deepseek.com` is reachable.
- `deepseek-chat` is usable and is the fixed model for the current 60-case run.
- `deepseek-v4-flash` can return empty `content` with reasoning-only output, so
  the provider now treats empty content as an error and falls back to the fixed
  content-producing model.

Remaining stability issue: official DeepSeek occasionally returned timeout or
503 busy responses during the 60-case run. Those cases were rerun with longer
timeouts before reporting `local_qa_v5_llm_merged.json`.
