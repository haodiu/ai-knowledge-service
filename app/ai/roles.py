"""The three LLM roles as services (Plan §8.2): build prompt -> call -> record -> return parsed.

This module owns the ONE retry/repair rule (Plan §11.7, CLAUDE.md invariant #7):
  * every provider call first takes a unit from the turn's pool (`budget.acquire_generative`);
  * a logical call gets at most ONE extra attempt — a *repair* after invalid output, or a *retry*
    after a transient error — and only if the pool still has a unit;
  * a retry never waits longer than 10 s, and never past the point where a call could still fit
    in the remaining deadline;
  * every attempt, failed or not, leaves one `model_calls` row (metadata only).
"""
import time
from collections.abc import Sequence

from pydantic import BaseModel

from app.ai.budget import TurnBudget
from app.ai.chat.base import ChatModelClient
from app.ai.embedding_recorder import EmbeddingCallRecord, EmbeddingCallRecorder
from app.ai.errors import (
    ModelError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
    StructuredOutputError,
)
from app.ai.prompts.builders import (
    build_answer_messages,
    build_grader_messages,
    build_planner_messages,
)
from app.ai.prompts.loader import Prompts
from app.ai.recorder import ModelCallRecord, ModelCallRecorder
from app.ai.registry import ModelRegistry
from app.ai.schemas import AnswerDraft, EvidenceGrade, QueryPlan
from app.ai.types import ModelMessage, Purpose
from app.retrieval.schemas import Evidence

RETRY_DEFAULT_DELAY_SECONDS = 2.0
RETRY_MAX_DELAY_SECONDS = 10.0
MIN_CALL_SECONDS = 3.0  # a retry must leave at least this much time for the call itself


def _transient_delay(exc: ModelError) -> float | None:
    """Seconds to wait before a retry, or None if `exc` is not worth retrying."""
    if isinstance(exc, ModelRateLimited):
        return exc.retry_after_seconds if exc.retry_after_seconds is not None \
            else RETRY_DEFAULT_DELAY_SECONDS
    if isinstance(exc, ModelTimeout):
        return 0.0  # the timeout already cost time; retry straight away if any is left
    if isinstance(exc, ModelUnavailable) and exc.retryable:
        return RETRY_DEFAULT_DELAY_SECONDS
    return None


def _affordable(delay: float, budget: TurnBudget) -> bool:
    fits_in_deadline = delay + MIN_CALL_SECONDS <= budget.remaining_seconds()
    return delay <= RETRY_MAX_DELAY_SECONDS and fits_in_deadline


def _repair_messages(
    messages: Sequence[ModelMessage], exc: StructuredOutputError
) -> list[ModelMessage]:
    fields = ", ".join(exc.fields) or "unknown"
    note = (
        "Your previous reply was rejected because it did not match the required JSON schema "
        f"(problem fields: {fields}). Reply again with only valid JSON that matches the schema."
    )
    return [*messages, ModelMessage("user", note)]  # locations only: never echo model output


async def _call[T: BaseModel](
    client: ChatModelClient,
    recorder: ModelCallRecorder,
    prompts: Prompts,
    budget: TurnBudget,
    *,
    purpose: Purpose,
    messages: Sequence[ModelMessage],
    response_model: type[T],
    timeout_seconds: float,
) -> T:
    attempt_messages: Sequence[ModelMessage] = messages
    attempt_purpose: Purpose = purpose
    extra_used = False
    while True:
        budget.acquire_generative()  # raises BEFORE any provider request is made
        started = time.perf_counter()
        try:
            response = await client.complete_structured(
                messages=attempt_messages,
                response_model=response_model,
                purpose=attempt_purpose,
                timeout_seconds=budget.call_timeout(timeout_seconds),
            )
        except ModelError as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            await recorder.record(
                ModelCallRecord(
                    attempt_purpose, client.provider, client.model_name, prompts.version,
                    None, None, elapsed, "error", exc.code,
                )
            )
            if extra_used or not budget.can_acquire_generative():
                raise
            if isinstance(exc, StructuredOutputError):
                if budget.remaining_seconds() < MIN_CALL_SECONDS:
                    raise
                attempt_messages = _repair_messages(messages, exc)
                attempt_purpose = "repair"
            else:
                delay = _transient_delay(exc)
                if delay is None or not _affordable(delay, budget):
                    raise
                if delay > 0:
                    await budget.sleep(delay)
            extra_used = True
            continue
        await recorder.record(
            ModelCallRecord(
                attempt_purpose, response.provider, response.model_name, prompts.version,
                response.usage.input_tokens, response.usage.output_tokens,
                response.latency_ms, "ok", None,
            )
        )
        return response.parsed


async def plan_query(
    models: ModelRegistry,
    recorder: ModelCallRecorder,
    prompts: Prompts,
    question: str,
    *,
    timeout_seconds: float,
    budget: TurnBudget,
    recent_turns: Sequence[str] = (),
) -> QueryPlan:
    return await _call(
        models.planner, recorder, prompts, budget,
        purpose="plan",
        messages=build_planner_messages(prompts, question, recent_turns=recent_turns),
        response_model=QueryPlan, timeout_seconds=timeout_seconds,
    )


async def grade_evidence(
    models: ModelRegistry,
    recorder: ModelCallRecorder,
    prompts: Prompts,
    question: str,
    evidence: Sequence[Evidence],
    *,
    timeout_seconds: float,
    budget: TurnBudget,
) -> EvidenceGrade:
    return await _call(
        models.grader, recorder, prompts, budget,
        purpose="grade", messages=build_grader_messages(prompts, question, evidence),
        response_model=EvidenceGrade, timeout_seconds=timeout_seconds,
    )


async def generate_answer(
    models: ModelRegistry,
    recorder: ModelCallRecorder,
    prompts: Prompts,
    question: str,
    evidence: Sequence[Evidence],
    *,
    timeout_seconds: float,
    budget: TurnBudget,
    recent_turns: Sequence[str] = (),
) -> AnswerDraft:
    return await _call(
        models.answer, recorder, prompts, budget,
        purpose="answer",
        messages=build_answer_messages(prompts, question, evidence, recent_turns=recent_turns),
        response_model=AnswerDraft, timeout_seconds=timeout_seconds,
    )


async def embed_query(
    models: ModelRegistry,
    recorder: EmbeddingCallRecorder,
    budget: TurnBudget,
    text: str,
    *,
    model_version: str,
) -> list[float]:
    """Embed the retrieval query. Not a generative call, so it does not use the pool, but it is
    bounded the same way: one transient retry, only if it fits in the remaining deadline. Every
    attempt, failed or not, leaves one embedding_calls row (Plan §17) -- the same "record every
    attempt" discipline `_call()` follows for chat calls."""
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            response = await models.embeddings.embed(
                [text], model_version=model_version, kind="query"
            )
        except ModelError as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            await recorder.record(
                EmbeddingCallRecord(
                    "query", models.embeddings.provider, model_version, 1, elapsed, "error",
                    exc.code,
                )
            )
            delay = _transient_delay(exc)
            if attempt == 2 or delay is None or not _affordable(delay, budget):
                raise
            if delay > 0:
                await budget.sleep(delay)
            continue
        elapsed = int((time.perf_counter() - started) * 1000)
        await recorder.record(
            EmbeddingCallRecord(
                "query", response.provider, response.model_version, 1, elapsed, "ok", None,
            )
        )
        return response.vectors[0]
    raise AssertionError("unreachable")  # pragma: no cover
