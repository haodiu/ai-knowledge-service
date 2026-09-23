"""Rate-limit policy interface (Plan §13.1). Kept separate from the Redis implementation so
callers/tests can fake it without a Redis client -- the same split as
`app.api.dependencies.Probe`.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: float | None = None


class RateLimiter(Protocol):
    async def check(self, user_id: str) -> RateLimitDecision: ...
