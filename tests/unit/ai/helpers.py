"""Shared builders for the Week 3 unit tests."""
import uuid

from app.retrieval.schemas import Evidence


def make_evidence(n: int = 2, *, text_prefix: str = "chunk") -> list[Evidence]:
    doc = uuid.uuid4()
    return [
        Evidence(
            chunk_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            document_id=doc,
            title=f"Doc {i}",
            version_no=1,
            chunk_index=i,
            text=f"{text_prefix} {i}",
            score=1.0 / (i + 1),
        )
        for i in range(n)
    ]


class FakeClock:
    """Deterministic time for budget/retry tests: `sleep` advances `now` instead of waiting."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_budget(clock: FakeClock, *, calls: int = 4, timeout: float = 30.0):  # type: ignore[no-untyped-def]
    from app.ai.budget import TurnBudget

    return TurnBudget.start(
        timeout_seconds=timeout, max_generative_calls=calls, max_tool_calls=1,
        clock=clock, sleep=clock.sleep,
    )
