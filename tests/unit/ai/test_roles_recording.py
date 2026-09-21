"""model_calls: purpose/model/prompt_version/tokens/latency for all 3 roles; never raw prompts."""
import dataclasses

import pytest

from app.ai.chat.fake import FakeChatModelClient
from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.errors import ModelRateLimited, StructuredOutputError
from app.ai.prompts.loader import load_prompts
from app.ai.recorder import InMemoryModelCallRecorder, ModelCallRecord
from app.ai.registry import ModelRegistry
from app.ai.roles import generate_answer, grade_evidence, plan_query
from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from tests.unit.ai.helpers import FakeClock, make_budget, make_evidence

PLAN = QueryPlan(intent="policy", retrieval_query="refund", needs_retrieval=True)
GRADE = EvidenceGrade(sufficient=True, confidence=0.9, reason="ok")


def _registry(planner=(), grader=(), answer=()) -> ModelRegistry:  # type: ignore[no-untyped-def]
    return ModelRegistry(
        planner=FakeChatModelClient(list(planner), provider="fakeprov", model_name="fake-plan"),
        grader=FakeChatModelClient(list(grader), provider="fakeprov", model_name="fake-grade"),
        answer=FakeChatModelClient(list(answer), provider="fakeprov", model_name="fake-answer"),
        embeddings=FakeEmbeddingClient(),
    )


def test_record_shape_has_no_place_for_prompt_or_response_text() -> None:
    fields = {f.name for f in dataclasses.fields(ModelCallRecord)}
    assert fields == {
        "purpose", "provider", "model_name", "prompt_version",
        "input_tokens", "output_tokens", "latency_ms", "status", "error_code",
    }


async def test_all_three_roles_are_recorded_with_their_own_model_and_prompt_version() -> None:
    ev = make_evidence(1)
    draft = AnswerDraft(
        status="answered", answer="a",
        citations=[CitationRef(document_version_id=ev[0].document_version_id,
                               chunk_id=ev[0].chunk_id)],
    )
    models = _registry([PLAN], [GRADE], [draft])
    rec = InMemoryModelCallRecorder()
    prompts = load_prompts("v1")

    await plan_query(models, rec, prompts, "q", timeout_seconds=5, budget=make_budget(FakeClock()))
    await grade_evidence(models, rec, prompts, "q", ev, timeout_seconds=5,
                         budget=make_budget(FakeClock()))
    await generate_answer(models, rec, prompts, "q", ev, timeout_seconds=5,
                          budget=make_budget(FakeClock()))

    assert [(r.purpose, r.model_name) for r in rec.records] == [
        ("plan", "fake-plan"), ("grade", "fake-grade"), ("answer", "fake-answer"),
    ]
    for r in rec.records:
        assert r.provider == "fakeprov" and r.prompt_version == "v1"
        assert r.status == "ok" and r.error_code is None
        assert r.latency_ms >= 0 and r.input_tokens and r.output_tokens


async def test_failure_is_recorded_with_a_normalised_error_code_and_reraised() -> None:
    """Week 4 policy: each failing call gets ONE extra attempt (repair / retry), and every attempt
    is recorded. (Week 3 scripted a single failure per role and expected it to stop there.)"""
    rate_limited = ModelRateLimited("x", retry_after_seconds=1)
    models = _registry(planner=["garbage", "garbage"], grader=[rate_limited, rate_limited])
    rec = InMemoryModelCallRecorder()
    prompts = load_prompts("v1")

    with pytest.raises(StructuredOutputError):
        await plan_query(models, rec, prompts, "q", timeout_seconds=5,
                         budget=make_budget(FakeClock()))
    with pytest.raises(ModelRateLimited):
        await grade_evidence(models, rec, prompts, "q", make_evidence(1), timeout_seconds=5,
                             budget=make_budget(FakeClock()))

    assert [(r.purpose, r.status, r.error_code) for r in rec.records] == [
        ("plan", "error", "structured_output_invalid"),
        ("repair", "error", "structured_output_invalid"),
        ("grade", "error", "rate_limited"),
        ("grade", "error", "rate_limited"),
    ]
    assert all(r.input_tokens is None and r.output_tokens is None for r in rec.records)
    assert all(r.provider == "fakeprov" for r in rec.records)


async def test_role_functions_pass_the_right_purpose_and_response_model() -> None:
    models = _registry([PLAN], [GRADE])
    rec = InMemoryModelCallRecorder()
    prompts = load_prompts("v1")
    await plan_query(models, rec, prompts, "q", timeout_seconds=5, budget=make_budget(FakeClock()))
    await grade_evidence(models, rec, prompts, "q", make_evidence(1), timeout_seconds=5,
                         budget=make_budget(FakeClock()))
    assert models.planner.calls[0].purpose == "plan"  # type: ignore[attr-defined]
    assert models.planner.calls[0].response_model is QueryPlan  # type: ignore[attr-defined]
    assert models.grader.calls[0].purpose == "grade"  # type: ignore[attr-defined]
    assert models.grader.calls[0].response_model is EvidenceGrade  # type: ignore[attr-defined]
    assert models.grader.calls[0].timeout_seconds == 5  # type: ignore[attr-defined]
