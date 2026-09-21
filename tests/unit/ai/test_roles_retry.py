"""Retry / repair policy in roles._call (Plan §11.7, CLAUDE.md invariant #7).

At most ONE extra attempt per logical call, every attempt consumes one unit of the turn pool,
every attempt leaves a model_calls row, and nothing sleeps past the remaining budget.
"""
import pytest

from app.ai.budget import BudgetExhausted
from app.ai.chat.fake import FakeChatModelClient
from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.errors import (
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
    StructuredOutputError,
)
from app.ai.prompts.loader import load_prompts
from app.ai.recorder import InMemoryModelCallRecorder
from app.ai.registry import ModelRegistry
from app.ai.roles import plan_query
from app.ai.schemas import QueryPlan
from tests.unit.ai.helpers import FakeClock, make_budget

PLAN = QueryPlan(intent="policy", retrieval_query="refund", needs_retrieval=True)
BAD = '{"intent": "policy"}'  # valid JSON, missing required fields


def _setup(script, *, calls: int = 4, timeout: float = 30.0):  # type: ignore[no-untyped-def]
    clock = FakeClock()
    planner = FakeChatModelClient(list(script))
    models = ModelRegistry(planner, FakeChatModelClient(), FakeChatModelClient(),
                           FakeEmbeddingClient())
    return clock, planner, models, make_budget(clock, calls=calls, timeout=timeout), \
        InMemoryModelCallRecorder()


async def _plan(models, rec, budget):  # type: ignore[no-untyped-def]
    return await plan_query(models, rec, load_prompts("v1"), "q", timeout_seconds=20.0,
                            budget=budget)


async def test_transient_error_is_retried_once_after_the_default_delay() -> None:
    clock, planner, models, budget, rec = _setup([ModelUnavailable("503"), PLAN])
    assert await _plan(models, rec, budget) == PLAN
    assert clock.sleeps == [2.0] and len(planner.calls) == 2
    assert [(r.purpose, r.status, r.error_code) for r in rec.records] == [
        ("plan", "error", "unavailable"), ("plan", "ok", None)]
    assert budget.generative_calls == 2


async def test_rate_limit_honours_retry_after_when_it_fits() -> None:
    clock, planner, models, budget, rec = _setup(
        [ModelRateLimited("x", retry_after_seconds=3.0), PLAN])
    await _plan(models, rec, budget)
    assert clock.sleeps == [3.0]


async def test_timeout_is_retried_without_waiting() -> None:
    clock, planner, models, budget, rec = _setup([ModelTimeout("slow"), PLAN])
    await _plan(models, rec, budget)
    assert clock.sleeps == [] and len(planner.calls) == 2


async def test_retry_after_longer_than_ten_seconds_is_not_waited_for() -> None:
    clock, planner, models, budget, rec = _setup(
        [ModelRateLimited("x", retry_after_seconds=11.0), PLAN])
    with pytest.raises(ModelRateLimited):
        await _plan(models, rec, budget)
    assert clock.sleeps == [] and len(planner.calls) == 1


async def test_no_retry_when_the_wait_would_not_leave_room_for_a_call() -> None:
    clock, planner, models, budget, rec = _setup(
        [ModelRateLimited("x", retry_after_seconds=5.0), PLAN], timeout=30.0)
    clock.now += 23.0  # 7 s left: 5 s wait + 3 s minimum call does not fit
    with pytest.raises(ModelRateLimited):
        await _plan(models, rec, budget)
    assert clock.sleeps == [] and len(planner.calls) == 1


async def test_non_retryable_provider_error_burns_no_extra_call() -> None:
    _, planner, models, budget, rec = _setup([ModelUnavailable("401", retryable=False), PLAN])
    with pytest.raises(ModelUnavailable):
        await _plan(models, rec, budget)
    assert len(planner.calls) == 1


async def test_only_one_extra_attempt_per_call() -> None:
    _, planner, models, budget, rec = _setup(
        [ModelUnavailable("503"), ModelUnavailable("503"), PLAN])
    with pytest.raises(ModelUnavailable):
        await _plan(models, rec, budget)
    assert len(planner.calls) == 2 and budget.generative_calls == 2


async def test_invalid_output_gets_exactly_one_repair_attempt() -> None:
    _, planner, models, budget, rec = _setup([BAD, PLAN])
    assert await _plan(models, rec, budget) == PLAN
    assert [r.purpose for r in rec.records] == ["plan", "repair"]
    assert [r.status for r in rec.records] == ["error", "ok"]
    assert planner.calls[1].purpose == "repair"
    assert planner.calls[1].response_model is QueryPlan  # same schema


async def test_repair_note_names_bad_fields_but_never_echoes_model_output() -> None:
    _, planner, models, budget, rec = _setup(
        ['{"intent": "policy", "note": "SECRET-INJECTED-TEXT"}', PLAN])
    await _plan(models, rec, budget)
    first, second = planner.calls[0].messages, planner.calls[1].messages
    assert second[:-1] == first and second[-1].role == "user"
    note = second[-1].content
    assert "retrieval_query" in note or "needs_retrieval" in note
    assert "SECRET-INJECTED-TEXT" not in note


async def test_a_second_invalid_output_is_final() -> None:
    _, planner, models, budget, rec = _setup([BAD, BAD, PLAN])
    with pytest.raises(StructuredOutputError):
        await _plan(models, rec, budget)
    assert len(planner.calls) == 2
    assert [r.purpose for r in rec.records] == ["plan", "repair"]


async def test_the_extra_attempt_is_one_total_even_across_error_kinds() -> None:
    _, planner, models, budget, rec = _setup([ModelUnavailable("503"), BAD, PLAN])
    with pytest.raises(StructuredOutputError):
        await _plan(models, rec, budget)
    assert len(planner.calls) == 2


async def test_no_extra_attempt_when_the_pool_has_no_unit_left() -> None:
    _, planner, models, budget, rec = _setup([ModelUnavailable("503"), PLAN], calls=1)
    with pytest.raises(ModelUnavailable):
        await _plan(models, rec, budget)
    assert len(planner.calls) == 1


async def test_exhausted_pool_fails_before_any_provider_call_and_records_nothing() -> None:
    _, planner, models, budget, rec = _setup([PLAN], calls=0)
    with pytest.raises(BudgetExhausted):
        await _plan(models, rec, budget)
    assert planner.calls == [] and rec.records == []


async def test_per_call_timeout_is_capped_by_the_remaining_deadline() -> None:
    clock, planner, models, budget, rec = _setup([PLAN], timeout=30.0)
    clock.now += 25.0
    await _plan(models, rec, budget)
    assert planner.calls[0].timeout_seconds == 5.0
