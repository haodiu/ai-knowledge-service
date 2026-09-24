"""Eval runner (Plan §16.4): drives `run_turn()` per golden question -- exactly like
`app/ai/cli.py`'s `ask` command, batched, not shelled out to. Scoring is deterministic set-
membership math, the same idea invariant #3's citation validator uses, applied against a golden
question's `expected_chunks`/`expected_citations` instead of the evidence set.

`answer_correctness`/`groundedness` are never computed here -- they stay blank for a human to
fill in (CLAUDE.md: "you build the runner, the human owns the gold answers"; no LLM-judge, per the
user's explicit choice). `tests/integration/test_eval_runner.py` asserts this as a guardrail.
"""
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.prompts.loader import Prompts
from app.ai.registry import ModelRegistry
from app.auth.policies import AuthorizationContext
from app.db import repositories
from app.eval.pricing import estimate_cost_usd
from app.eval.schema import ExpectedChunk, GoldenQuestion
from app.graph.limits import MAX_GENERATIVE_LLM_CALLS
from app.graph.result import TurnResult
from app.graph.runner import run_turn
from app.tools.subscription import SubscriptionToolClient

# (str(document_version_id), str(chunk_id)) -- hashable, JSON-round-trippable.
ChunkKey = tuple[str, str]


@dataclass(frozen=True)
class QuestionResult:
    question: GoldenQuestion
    turn_result: TurnResult
    latency_ms: int
    calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    retrieval_recall_at_k: float | None
    citation_precision: float | None
    citation_recall: float | None
    status_matches_expected: bool
    tool_routing_correct: bool
    exceeds_call_budget: bool
    resolution_errors: tuple[str, ...] = ()
    # Human-fill ONLY -- the runner must never populate these. See this module's docstring.
    answer_correctness: str = ""
    groundedness: str = ""


async def resolve_expected_chunks(
    engine: AsyncEngine, chunks: Sequence[ExpectedChunk]
) -> tuple[dict[ExpectedChunk, ChunkKey], tuple[str, ...]]:
    """Resolve each (external_id, chunk_index) to the active version's real
    (document_version_id, chunk_id) -- unknowable at authoring time (Plan §16.4, see
    app/eval/schema.py). Same two-condition active-version join CLAUDE.md's invariant #2 requires
    of the real retrieval path (dv.id = d.active_version_id AND dv.document_id = d.id), reused
    here for the same reason: a mis-pointed document must not silently resolve to the wrong
    version. An unresolvable reference is reported, never silently dropped."""
    resolved: dict[ExpectedChunk, ChunkKey] = {}
    errors: list[str] = []
    async with engine.connect() as conn:
        for ec in chunks:
            row = (
                await conn.execute(
                    text(
                        "SELECT dv.id AS version_id, c.id AS chunk_id "
                        "FROM documents d "
                        "JOIN document_versions dv "
                        "  ON dv.id = d.active_version_id AND dv.document_id = d.id "
                        "JOIN chunks c ON c.document_version_id = dv.id "
                        "WHERE d.external_id = :e AND c.chunk_index = :i"
                    ),
                    {"e": ec.external_id, "i": ec.chunk_index},
                )
            ).one_or_none()
            if row is None:
                errors.append(f"{ec.external_id}#{ec.chunk_index}: not found in the active version")
                continue
            resolved[ec] = (str(row.version_id), str(row.chunk_id))
    return resolved, tuple(errors)


def _precision(expected: set[ChunkKey], actual: set[ChunkKey]) -> float | None:
    return len(expected & actual) / len(actual) if actual else None


def _recall(expected: set[ChunkKey], actual: set[ChunkKey]) -> float | None:
    return len(expected & actual) / len(expected) if expected else None


async def _score_one(
    engine: AsyncEngine,
    question: GoldenQuestion,
    result: TurnResult,
    *,
    latency_ms: int,
    calls: int,
    input_tokens: int,
    output_tokens: int,
    answer_model_name: str,
) -> QuestionResult:
    expected_chunks, chunk_errors = await resolve_expected_chunks(engine, question.expected_chunks)
    expected_citations, citation_errors = await resolve_expected_chunks(
        engine, question.expected_citations
    )
    expected_evidence_set = set(expected_chunks.values())
    expected_citation_set = set(expected_citations.values())
    actual_evidence_set = {(str(e.document_version_id), str(e.chunk_id)) for e in result.evidence}
    actual_citation_set = {(str(c.document_version_id), str(c.chunk_id)) for c in result.citations}

    return QuestionResult(
        question=question,
        turn_result=result,
        latency_ms=latency_ms,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost_usd(answer_model_name, input_tokens, output_tokens),
        retrieval_recall_at_k=_recall(expected_evidence_set, actual_evidence_set),
        citation_precision=_precision(expected_citation_set, actual_citation_set),
        citation_recall=_recall(expected_citation_set, actual_citation_set),
        status_matches_expected=result.status == question.expected_type,
        tool_routing_correct=(
            (result.proposed_tool.name if result.proposed_tool else None) == question.expected_tool
        ),
        exceeds_call_budget=calls > MAX_GENERATIVE_LLM_CALLS,
        resolution_errors=chunk_errors + citation_errors,
    )


async def run_eval(
    engine: AsyncEngine,
    models: ModelRegistry,
    prompts: Prompts,
    tool_client: SubscriptionToolClient | None,
    questions: Sequence[GoldenQuestion],
    *,
    embedding_model: str,
    chat_timeout_seconds: float,
    tool_timeout_seconds: float,
    user_id: str = "eval",
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for question in questions:
        created = await repositories.create_turn(
            engine, user_id=user_id, question=question.question
        )
        started = time.perf_counter()
        result = await run_turn(
            engine=engine,
            conversation_id=created.conversation_id,
            turn_id=created.turn_id,
            question=question.question,
            auth=AuthorizationContext(user_id=user_id, tier=question.tier),
            models=models,
            prompts=prompts,
            embedding_model=embedding_model,
            chat_timeout_seconds=chat_timeout_seconds,
            tool_client=tool_client,
            tool_timeout_seconds=tool_timeout_seconds,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        calls, input_tokens, output_tokens = await repositories.get_turn_usage(
            engine, created.turn_id
        )
        results.append(
            await _score_one(
                engine, question, result,
                latency_ms=latency_ms, calls=calls,
                input_tokens=input_tokens, output_tokens=output_tokens,
                answer_model_name=models.answer.model_name,
            )
        )
    return results
