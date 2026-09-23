"""TurnBudget: the mechanism behind MAX_GENERATIVE_LLM_CALLS, MAX_TOOL_CALLS and the deadline.

Request-scoped and mutable, so it lives in the runtime context, never in graph state. Every
provider call — first attempt, retry, repair — must `acquire_generative()` FIRST; the acquire
raises before any request is made, so the limit cannot be exceeded by a code path that forgot to
count (invariant #7). Limits are passed in by the graph runner from `graph/limits.py`.
"""
import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.ai.errors import ModelError


class BudgetExhausted(ModelError):
    code = "budget_exhausted"


@dataclass
class TurnBudget:
    max_generative_calls: int
    max_tool_calls: int
    deadline: float
    clock: Callable[[], float]
    sleep: Callable[[float], Awaitable[None]]
    generative_calls: int = 0
    tool_calls: int = 0

    @classmethod
    def start(
        cls,
        *,
        timeout_seconds: float,
        max_generative_calls: int,
        max_tool_calls: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> "TurnBudget":
        return cls(max_generative_calls, max_tool_calls, clock() + timeout_seconds, clock, sleep)

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - self.clock())

    def can_acquire_generative(self) -> bool:
        return self.generative_calls < self.max_generative_calls and self.remaining_seconds() > 0

    def acquire_generative(self) -> None:
        if not self.can_acquire_generative():
            raise BudgetExhausted("generative call budget or deadline exhausted")
        self.generative_calls += 1

    def can_acquire_tool(self) -> bool:
        return self.tool_calls < self.max_tool_calls and self.remaining_seconds() > 0

    def acquire_tool(self) -> None:
        if not self.can_acquire_tool():
            raise BudgetExhausted("tool call budget or deadline exhausted")
        self.tool_calls += 1

    def call_timeout(self, configured: float) -> float:
        return min(configured, self.remaining_seconds())
