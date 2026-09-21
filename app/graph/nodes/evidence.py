"""build_evidence (deterministic) and grade_evidence (LLM #2)."""
from collections.abc import Sequence
from typing import Any

from app.ai.errors import ModelError
from app.ai.roles import grade_evidence
from app.graph.limits import MAX_EVIDENCE_CHARS, MAX_RETRIEVED_CHUNKS
from app.graph.nodes._errors import to_turn_error
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState
from app.retrieval.schemas import Evidence


def evidence_key(e: Evidence) -> tuple[object, object]:
    return (e.document_version_id, e.chunk_id)  # the citation key (Plan §9.3)


def build_evidence(
    existing: Sequence[Evidence],
    new: Sequence[Evidence],
    *,
    max_chunks: int = MAX_RETRIEVED_CHUNKS,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> list[Evidence]:
    """Union (deduped), capped by count and by characters; NEW results take priority.

    RRF scores are ranks within ONE query, so scores from the original and the rewritten query are
    not comparable. Ranking the union by score could push out the very chunks the rewrite just
    found, so `new` (best first) is placed ahead of `existing` (best first) and the cap trims the
    older evidence. On the first retrieval `existing` is empty and this is plain best-first.

    A chunk is kept whole or dropped whole — never cut — so a source header can never be separated
    from its text (Plan §11.5). The top chunk is always kept, even if it alone exceeds the budget.
    """
    def best_first(items: Sequence[Evidence]) -> list[Evidence]:
        return sorted(items, key=lambda e: (-e.score, str(e.chunk_id)))

    unique: dict[tuple[object, object], Evidence] = {}
    for e in [*best_first(new), *best_first(existing)]:
        unique.setdefault(evidence_key(e), e)
    ranked = list(unique.values())

    kept: list[Evidence] = []
    used = 0
    for e in ranked:
        if len(kept) >= max_chunks:
            break
        if kept and used + len(e.text) > max_chars:
            break
        kept.append(e)
        used += len(e.text)
    return kept


async def grade_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    await ctx.emit("grading")
    try:
        grade = await grade_evidence(
            ctx.models, ctx.recorder, ctx.prompts, state["normalized_question"],
            state["evidence"], timeout_seconds=ctx.chat_timeout_seconds, budget=ctx.budget,
        )
    except ModelError as exc:
        return {"error": to_turn_error(exc)}
    return {"grade": grade}


async def rewrite_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    """Deterministic: adopt the grader's rewrite. The router already validated it, and this node
    opens no new loop — the next stop is the same bounded retrieve -> grade path."""
    grade = state["grade"]
    assert grade is not None and grade.rewritten_query is not None
    return {"retrieval_query": grade.rewritten_query.strip()}
