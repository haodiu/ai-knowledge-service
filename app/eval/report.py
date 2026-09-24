"""Eval report (Plan §16.4): per-question rows + an aggregate summary, written as JSON (machine-
readable) and Markdown (human-readable), with the correctness/groundedness columns visibly TODO.
"""
import json
from pathlib import Path
from typing import cast

from app.eval.runner import QuestionResult


def _question_row(r: QuestionResult) -> dict[str, object]:
    return {
        "id": r.question.id,
        "category_notes": r.question.notes,
        "question": r.question.question,
        "tier": r.question.tier.value,
        "expected_type": r.question.expected_type,
        "actual_status": r.turn_result.status,
        "actual_detail": r.turn_result.detail,
        "status_matches_expected": r.status_matches_expected,
        "retrieval_recall_at_k": r.retrieval_recall_at_k,
        "citation_precision": r.citation_precision,
        "citation_recall": r.citation_recall,
        "expected_tool": r.question.expected_tool,
        "tool_routing_correct": r.tool_routing_correct,
        "generative_calls": r.calls,
        "exceeds_call_budget": r.exceeds_call_budget,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "estimated_cost_usd": r.estimated_cost_usd,
        "latency_ms": r.latency_ms,
        "resolution_errors": list(r.resolution_errors),
        "answer": r.turn_result.answer,
        "answer_correctness": r.answer_correctness,  # TODO(human)
        "groundedness": r.groundedness,  # TODO(human)
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _p95(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return float(ordered[index])


def _insufficient_evidence_precision(results: list[QuestionResult]) -> float | None:
    """Aggregate over the whole run, not per-question (Plan §16.4's "insufficient-evidence
    precision" is a set-level metric: of everything the system called insufficient_evidence, how
    much genuinely was)."""
    predicted_insufficient = [r for r in results if r.turn_result.status == "insufficient_evidence"]
    if not predicted_insufficient:
        return None
    true_positives = sum(
        1 for r in predicted_insufficient if r.question.expected_type == "insufficient_evidence"
    )
    return true_positives / len(predicted_insufficient)


def summarize(results: list[QuestionResult]) -> dict[str, object]:
    recalls = [r.retrieval_recall_at_k for r in results if r.retrieval_recall_at_k is not None]
    precisions = [r.citation_precision for r in results if r.citation_precision is not None]
    citation_recalls = [r.citation_recall for r in results if r.citation_recall is not None]
    costs = [r.estimated_cost_usd for r in results if r.estimated_cost_usd is not None]
    return {
        "question_count": len(results),
        "status_match_rate": _mean([1.0 if r.status_matches_expected else 0.0 for r in results]),
        "tool_routing_accuracy": _mean([1.0 if r.tool_routing_correct else 0.0 for r in results]),
        "mean_retrieval_recall_at_k": _mean(recalls),
        "mean_citation_precision": _mean(precisions),
        "mean_citation_recall": _mean(citation_recalls),
        "insufficient_evidence_precision": _insufficient_evidence_precision(results),
        "mean_generative_calls": _mean([float(r.calls) for r in results]),
        "any_call_budget_exceeded": any(r.exceeds_call_budget for r in results),
        "mean_input_tokens": _mean([float(r.input_tokens) for r in results]),
        "mean_output_tokens": _mean([float(r.output_tokens) for r in results]),
        "mean_estimated_cost_usd": _mean(costs),
        "total_estimated_cost_usd": sum(costs) if costs else None,
        "mean_latency_ms": _mean([float(r.latency_ms) for r in results]),
        "p95_latency_ms": _p95([r.latency_ms for r in results]),
        "questions_with_unresolved_expectations": [
            r.question.id for r in results if r.resolution_errors
        ],
    }


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _markdown_table(results: list[QuestionResult]) -> str:
    headers = [
        "id", "expected", "actual", "match", "recall@k", "cite P", "cite R", "tool", "calls",
        "tokens", "cost $", "latency ms", "correctness (TODO)", "groundedness (TODO)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in results:
        row = [
            r.question.id,
            r.question.expected_type,
            r.turn_result.status,
            "✓" if r.status_matches_expected else "✗",
            _fmt(r.retrieval_recall_at_k),
            _fmt(r.citation_precision),
            _fmt(r.citation_recall),
            "✓" if r.tool_routing_correct else "✗",
            str(r.calls),
            str(r.input_tokens + r.output_tokens),
            _fmt(r.estimated_cost_usd),
            str(r.latency_ms),
            r.answer_correctness or "TODO",
            r.groundedness or "TODO",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(results: list[QuestionResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)

    (out_dir / "results.json").write_text(
        json.dumps(
            {"summary": summary, "questions": [_question_row(r) for r in results]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Eval report",
        "",
        f"{summary['question_count']} questions. "
        "**`correctness`/`groundedness` are human-graded -- always blank here by design "
        "(no LLM-judge; see app/eval/runner.py).**",
        "",
        "## Summary",
        "",
        *(
            f"- **{k}**: {_fmt(v)}"
            for k, v in summary.items()
            if k != "questions_with_unresolved_expectations"
        ),
    ]
    unresolved = cast(list[str], summary["questions_with_unresolved_expectations"])
    if unresolved:
        lines.append(
            f"- ⚠️ **unresolved expected chunks/citations** in: {', '.join(unresolved)} "
            "(the corpus may not be ingested, or a golden question references a stale chunk_index)"
        )
    lines += ["", "## Per-question results", "", _markdown_table(results), ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
