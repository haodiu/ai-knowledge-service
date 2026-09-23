"""RedisRateLimiter (Plan §13.1, §13.3). A tiny in-memory fake stands in for redis.asyncio.Redis
-- only `incr`/`expire` are used, and a third fake that raises RedisError proves fail-open/closed.
"""
import pytest
from redis.exceptions import RedisError

from app.rate_limit.redis import RateLimitUnavailable, RedisRateLimiter


class FakeRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.expired: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, seconds: int) -> None:
        self.expired[key] = seconds


class BrokenRedis:
    async def incr(self, key: str) -> int:
        raise RedisError("connection refused")

    async def expire(self, key: str, seconds: int) -> None:
        raise RedisError("connection refused")


async def test_requests_within_quota_are_allowed() -> None:
    limiter = RedisRateLimiter(FakeRedis(), requests_per_window=3, fail_open=True)  # type: ignore[arg-type]
    for _ in range(3):
        decision = await limiter.check("user-1")
        assert decision.allowed


async def test_request_beyond_quota_is_rejected_with_retry_after() -> None:
    limiter = RedisRateLimiter(FakeRedis(), requests_per_window=2, fail_open=True)  # type: ignore[arg-type]
    await limiter.check("user-1")
    await limiter.check("user-1")
    decision = await limiter.check("user-1")
    assert not decision.allowed
    assert decision.retry_after_seconds is not None
    assert 0 < decision.retry_after_seconds <= 60


async def test_quota_is_per_user() -> None:
    limiter = RedisRateLimiter(FakeRedis(), requests_per_window=1, fail_open=True)  # type: ignore[arg-type]
    assert (await limiter.check("user-1")).allowed
    assert (await limiter.check("user-2")).allowed  # different user, own quota


async def test_expire_set_only_on_first_request_in_window() -> None:
    redis = FakeRedis()
    limiter = RedisRateLimiter(redis, requests_per_window=5, fail_open=True)  # type: ignore[arg-type]
    await limiter.check("user-1")
    await limiter.check("user-1")
    assert len(redis.expired) == 1  # EXPIRE called once, not on every increment


async def test_redis_unavailable_fails_open_by_default() -> None:
    limiter = RedisRateLimiter(BrokenRedis(), requests_per_window=1, fail_open=True)  # type: ignore[arg-type]
    decision = await limiter.check("user-1")
    assert decision.allowed


async def test_redis_unavailable_fails_closed_when_configured() -> None:
    limiter = RedisRateLimiter(BrokenRedis(), requests_per_window=1, fail_open=False)  # type: ignore[arg-type]
    with pytest.raises(RateLimitUnavailable):
        await limiter.check("user-1")
