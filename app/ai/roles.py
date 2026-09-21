"""The three LLM roles as services (Plan §8.2): build prompt -> call -> record -> return parsed.

Week 4's LangGraph nodes call these; they do no routing, no retries and no budget accounting.
Every attempt — success or failure — leaves one `model_calls` record (metadata only).
"""
import time
from collections.abc import Sequence

from pydantic import BaseModel

from app.ai.chat.base import ChatModelClient
from app.ai.errors import ModelError
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


async def _call[T: BaseModel](
    client: ChatModelClient,
    recorder: ModelCallRecorder,
    prompts: Prompts,
    *,
    purpose: Purpose,
    messages: Sequence[ModelMessage],
    response_model: type[T],
    timeout_seconds: float,
) -> T:
    started = time.perf_counter()
    try:
        response = await client.complete_structured(
            messages=messages,
            response_model=response_model,
            purpose=purpose,
            timeout_seconds=timeout_seconds,
        )
    except ModelError as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        await recorder.record(
            ModelCallRecord(
                purpose, client.provider, client.model_name, prompts.version,
                None, None, elapsed, "error", exc.code,
            )
        )
        raise
    await recorder.record(
        ModelCallRecord(
            purpose, response.provider, response.model_name, prompts.version,
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
) -> QueryPlan:
    return await _call(
        models.planner, recorder, prompts,
        purpose="plan", messages=build_planner_messages(prompts, question),
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
) -> EvidenceGrade:
    return await _call(
        models.grader, recorder, prompts,
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
) -> AnswerDraft:
    return await _call(
        models.answer, recorder, prompts,
        purpose="answer", messages=build_answer_messages(prompts, question, evidence),
        response_model=AnswerDraft, timeout_seconds=timeout_seconds,
    )
