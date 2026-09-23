"""Run one turn: deadline + graph + atomic persistence.

`execute_turn` is the DB-free core (unit-testable with fakes). `run_turn` adds the request wiring
and persists the turn together with its citation snapshots in ONE transaction, BEFORE returning —
so whatever streams the answer later (Week 6 SSE) can only follow a persisted, validated turn.
"""
import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.budget import TurnBudget
from app.ai.prompts.loader import Prompts
from app.ai.recorder import SqlModelCallRecorder
from app.ai.registry import ModelRegistry
from app.auth.policies import AuthorizationContext
from app.db import repositories
from app.graph.limits import (
    GRAPH_RECURSION_LIMIT,
    GRAPH_TIMEOUT_SECONDS,
    MAX_GENERATIVE_LLM_CALLS,
    MAX_HISTORY_CHARS_PER_TURN,
    MAX_HISTORY_TURNS,
    MAX_RETRIEVED_CHUNKS,
    MAX_TOOL_CALLS,
)
from app.graph.result import KNOWN_DETAILS, TurnResult, TurnStatus
from app.graph.runtime import GraphRuntimeContext, PhaseCallback, Retriever
from app.graph.state import RAGState
from app.graph.workflow import GraphLoopError, run_graph
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.schemas import Evidence, Tier
from app.tools.recorder import SqlToolCallRecorder
from app.tools.subscription import SubscriptionToolClient

log = logging.getLogger(__name__)


async def invoke_graph(
    question: str,
    ctx: GraphRuntimeContext,
    *,
    timeout_seconds: float = GRAPH_TIMEOUT_SECONDS,
    recent_turns: Sequence[str] = (),
) -> RAGState:
    async with asyncio.timeout(timeout_seconds):  # invariant #7: GRAPH_TIMEOUT_SECONDS
        return await run_graph(
            question, ctx, recursion_limit=GRAPH_RECURSION_LIMIT, recent_turns=recent_turns
        )


def _failed(status: TurnStatus, detail: str) -> TurnResult:
    return TurnResult(status=status, detail=detail)


async def execute_turn(
    question: str,
    ctx: GraphRuntimeContext,
    *,
    timeout_seconds: float = GRAPH_TIMEOUT_SECONDS,
    recent_turns: Sequence[str] = (),
) -> TurnResult:
    try:
        state = await invoke_graph(
            question, ctx, timeout_seconds=timeout_seconds, recent_turns=recent_turns
        )
    except TimeoutError:
        return _failed("temporarily_unavailable", "graph_timeout")
    except GraphLoopError:  # unreachable while the routing bounds hold; fail closed anyway
        return _failed("blocked", "graph_recursion_limit")
    except Exception:
        log.exception("turn %s failed unexpectedly", ctx.request_id)
        return _failed("temporarily_unavailable", "internal_error")

    status: TurnStatus = state.get("status", "blocked")  # type: ignore[assignment]
    detail = state.get("detail", "")
    if detail not in KNOWN_DETAILS or (status == "answered") != (detail == "ok"):
        # an outcome nobody named (or an "answered" without the validated-answer detail): fail
        # closed and say so loudly instead of returning whatever the state happens to hold
        log.error("turn %s ended in an unnamed state: status=%r detail=%r",
                  ctx.request_id, status, detail)
        return _failed("blocked", "unexpected_state")
    answered = status == "answered"
    return TurnResult(
        status=status,
        answer=state.get("answer") if answered else None,
        clarification_question=state.get("clarification_question"),
        citations=tuple(state.get("citations", ())) if answered else (),
        evidence=tuple(state.get("evidence", ())),
        plan=state.get("plan"),
        grade=state.get("grade"),
        proposed_tool=state.get("proposed_tool"),
        tool_executed=state.get("tool_executed", False),
        retry_after_seconds=state.get("retry_after_seconds"),
        detail=detail,
        retrieval_attempts=state.get("retrieval_attempts", 0),
        source_snapshots=tuple(state.get("source_snapshots", ())) if answered else (),
    )


def _default_retriever(engine: AsyncEngine) -> Retriever:
    async def retrieve(
        query: str, embedding: Sequence[float], allowed_tiers: Sequence[Tier]
    ) -> Sequence[Evidence]:
        return await hybrid_search(
            engine, query_text=query, query_embedding=embedding,
            allowed_tiers=allowed_tiers, limit=MAX_RETRIEVED_CHUNKS,
        )

    return retrieve


async def run_turn(
    *,
    engine: AsyncEngine,
    conversation_id: uuid.UUID,
    turn_id: uuid.UUID,
    question: str,
    auth: AuthorizationContext,
    models: ModelRegistry,
    prompts: Prompts,
    embedding_model: str,
    chat_timeout_seconds: float,
    tool_client: SubscriptionToolClient | None,
    tool_timeout_seconds: float,
    retriever: Retriever | None = None,
    on_phase: PhaseCallback | None = None,
    timeout_seconds: float = GRAPH_TIMEOUT_SECONDS,
) -> TurnResult:
    started = time.perf_counter()
    recent_turns = await repositories.get_recent_turns(
        engine, conversation_id,
        limit=MAX_HISTORY_TURNS, max_chars_per_turn=MAX_HISTORY_CHARS_PER_TURN,
    )
    ctx = GraphRuntimeContext(
        request_id=str(turn_id),
        auth=auth,
        retriever=retriever or _default_retriever(engine),
        models=models,
        recorder=SqlModelCallRecorder(engine, turn_id),
        prompts=prompts,
        budget=TurnBudget.start(
            timeout_seconds=timeout_seconds,
            max_generative_calls=MAX_GENERATIVE_LLM_CALLS,
            max_tool_calls=MAX_TOOL_CALLS,
        ),
        embedding_model=embedding_model,
        chat_timeout_seconds=chat_timeout_seconds,
        tool_client=tool_client,
        tool_recorder=SqlToolCallRecorder(engine, turn_id),
        tool_timeout_seconds=tool_timeout_seconds,
        on_phase=on_phase,
    )
    result = await execute_turn(
        question, ctx, timeout_seconds=timeout_seconds, recent_turns=recent_turns
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    try:
        await repositories.finish_turn(
            engine, turn_id,
            graph_status=result.status, answer=result.answer,
            sources=[{"document_version_id": str(c.document_version_id),
                      "chunk_id": str(c.chunk_id)} for c in result.citations],
            retrieval_attempts=result.retrieval_attempts, latency_ms=latency_ms,
            snapshots=result.source_snapshots,
        )
    except Exception:
        # An answer whose citations cannot be opened later must not be returned: fail closed.
        log.exception("turn %s: persisting the turn/snapshots failed", turn_id)
        result = replace(
            result, status="temporarily_unavailable", detail="persistence_failed", answer=None,
            clarification_question=None, citations=(), source_snapshots=(),
        )
        try:
            await repositories.finish_turn(
                engine, turn_id, graph_status=result.status, answer=None, sources=[],
                retrieval_attempts=result.retrieval_attempts, latency_ms=latency_ms,
            )
        except Exception:
            log.exception("turn %s: could not record the failure either", turn_id)

    # Token/cost observability (Plan §17): one summary line per turn, correlated by turn_id (the
    # request_id). No new table -- this sums the model_calls rows insert_model_call() already
    # wrote for this turn. Best effort: a failure here must not affect the turn's own outcome.
    try:
        calls, input_tokens, output_tokens = await repositories.get_turn_usage(engine, turn_id)
        log.info(
            "turn summary: request_id=%s conversation_id=%s status=%s detail=%s "
            "retrieval_attempts=%d latency_ms=%d generative_calls=%d "
            "input_tokens=%d output_tokens=%d",
            turn_id, conversation_id, result.status, result.detail, result.retrieval_attempts,
            latency_ms, calls, input_tokens, output_tokens,
        )
    except Exception:
        log.warning("turn %s: could not summarise token usage", turn_id, exc_info=True)
    return result
