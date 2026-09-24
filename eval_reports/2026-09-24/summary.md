# Eval report

22 questions. **`correctness`/`groundedness` are human-graded -- always blank here by design (no LLM-judge; see app/eval/runner.py).**

## Summary

- **question_count**: 22
- **status_match_rate**: 0.727
- **tool_routing_accuracy**: 1
- **mean_retrieval_recall_at_k**: 1
- **mean_citation_precision**: 1
- **mean_citation_recall**: 1
- **insufficient_evidence_precision**: 1
- **mean_generative_calls**: 2.55
- **any_call_budget_exceeded**: False
- **mean_input_tokens**: 2.23e+03
- **mean_output_tokens**: 179
- **mean_estimated_cost_usd**: 0.000295
- **total_estimated_cost_usd**: 0.00648
- **mean_latency_ms**: 8.13e+03
- **p95_latency_ms**: 1.38e+04

## Per-question results

| id | expected | actual | match | recall@k | cite P | cite R | tool | calls | tokens | cost $ | latency ms | correctness (TODO) | groundedness (TODO) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| docs-01 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 4 | 3635 | 0.000428 | 14035 | TODO | TODO |
| docs-02 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 3 | 3634 | 0.000432 | 13079 | TODO | TODO |
| docs-03 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 3 | 3620 | 0.000429 | 10392 | TODO | TODO |
| docs-04 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 3 | 3667 | 0.000448 | 8980 | TODO | TODO |
| docs-05 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 4 | 3684 | 0.000443 | 13735 | TODO | TODO |
| docs-06 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 3 | 3602 | 0.000433 | 13669 | TODO | TODO |
| docs-07 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 4 | 3621 | 0.00043 | 12537 | TODO | TODO |
| multi-01 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 4 | 3869 | 0.000518 | 13813 | TODO | TODO |
| multi-02 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 3 | 3764 | 0.000476 | 8681 | TODO | TODO |
| multi-03 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 3 | 3816 | 0.000496 | 10822 | TODO | TODO |
| kb-01 | insufficient_evidence | insufficient_evidence | ✓ |  |  |  | ✓ | 2 | 2045 | 0.000244 | 8615 | TODO | TODO |
| kb-02 | insufficient_evidence | insufficient_evidence | ✓ |  |  |  | ✓ | 2 | 2049 | 0.000244 | 6126 | TODO | TODO |
| kb-03 | insufficient_evidence | insufficient_evidence | ✓ |  |  |  | ✓ | 2 | 2053 | 0.000244 | 5711 | TODO | TODO |
| internal-01 | insufficient_evidence | insufficient_evidence | ✓ |  |  |  | ✓ | 2 | 2026 | 0.000245 | 6027 | TODO | TODO |
| internal-02 | insufficient_evidence | insufficient_evidence | ✓ |  |  |  | ✓ | 2 | 2044 | 0.000243 | 6578 | TODO | TODO |
| internal-03 | answered | answered | ✓ | 1 | 1 | 1 | ✓ | 3 | 4145 | 0.000482 | 9757 | TODO | TODO |
| tool-01 | answered | temporarily_unavailable | ✗ |  |  |  | ✓ | 1 | 576 | 8.31e-05 | 2950 | TODO | TODO |
| tool-02 | answered | temporarily_unavailable | ✗ |  |  |  | ✓ | 1 | 578 | 8.3e-05 | 3516 | TODO | TODO |
| tool-03 | answered | temporarily_unavailable | ✗ |  |  |  | ✓ | 1 | 580 | 8.38e-05 | 2135 | TODO | TODO |
| ambiguous-01 | clarification | temporarily_unavailable | ✗ |  |  |  | ✓ | 2 | 0 | 0 | 2504 | TODO | TODO |
| ambiguous-02 | clarification | temporarily_unavailable | ✗ |  |  |  | ✓ | 2 | 0 | 0 | 2547 | TODO | TODO |
| ambiguous-03 | clarification | temporarily_unavailable | ✗ |  |  |  | ✓ | 2 | 0 | 0 | 2545 | TODO | TODO |
