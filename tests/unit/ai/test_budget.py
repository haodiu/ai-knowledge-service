"""TurnBudget: the mechanism behind MAX_GENERATIVE_LLM_CALLS / MAX_TOOL_CALLS / the deadline."""
import pytest

from app.ai.budget import BudgetExhausted
from tests.unit.ai.helpers import FakeClock, make_budget


def test_generative_calls_are_capped_and_the_cap_raises_before_a_call_is_made() -> None:
    budget = make_budget(FakeClock(), calls=4)
    for _ in range(4):
        budget.acquire_generative()
    assert not budget.can_acquire_generative()
    with pytest.raises(BudgetExhausted) as exc:
        budget.acquire_generative()
    assert exc.value.code == "budget_exhausted"
    assert budget.generative_calls == 4  # the failed acquire did not count


def test_tool_calls_are_capped_at_one() -> None:
    budget = make_budget(FakeClock())
    budget.acquire_tool()
    with pytest.raises(BudgetExhausted):
        budget.acquire_tool()


def test_remaining_time_follows_the_clock_and_never_goes_negative() -> None:
    clock = FakeClock()
    budget = make_budget(clock, timeout=30.0)
    assert budget.remaining_seconds() == 30.0
    clock.now += 12.5
    assert budget.remaining_seconds() == 17.5
    clock.now += 100
    assert budget.remaining_seconds() == 0.0


def test_no_call_can_start_after_the_deadline() -> None:
    clock = FakeClock()
    budget = make_budget(clock, timeout=30.0)
    clock.now += 30.0
    assert not budget.can_acquire_generative()
    with pytest.raises(BudgetExhausted):
        budget.acquire_generative()


def test_no_tool_call_can_start_after_the_deadline_either() -> None:
    """acquire_tool() must be bounded by the deadline the same way acquire_generative() is --
    otherwise a tool call could be "acquired" with ~0s left, and `call_timeout()` would then hand
    httpx a timeout of exactly 0, which httpx treats as NO timeout, not "expire immediately"."""
    clock = FakeClock()
    budget = make_budget(clock, timeout=30.0)
    clock.now += 30.0
    assert not budget.can_acquire_tool()
    with pytest.raises(BudgetExhausted):
        budget.acquire_tool()
    assert budget.tool_calls == 0  # the failed acquire did not count


def test_per_call_timeout_is_the_smaller_of_configured_and_remaining() -> None:
    clock = FakeClock()
    budget = make_budget(clock, timeout=30.0)
    assert budget.call_timeout(20.0) == 20.0
    clock.now += 25.0
    assert budget.call_timeout(20.0) == 5.0
