"""Fixed-window rate limiter on Redis (Plan §13.1, §13.3).

Key: `rag:ratelimit:user:{user_id}:{window}` -- exactly the prefix Plan §13.1 specifies, with a
window suffix appended. `window` is the current 60s bucket (epoch // 60), so the counter resets
cleanly every minute without a separate cleanup job: an untouched key just expires on its own.

Fail-open by default (Week 6 decision): Redis here is an abuse-prevention control, not
authorization -- JWT already gates access (app/auth/jwt.py). A Redis outage must not take chat
down. `fail_open=False` is available for environments that want the stricter trade-off instead.
"""
import logging
import time

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.rate_limit.policy import RateLimitDecision

_LOG = logging.getLogger(__name__)
_WINDOW_SECONDS = 60


class RateLimitUnavailable(Exception):
    """Redis could not be reached and `fail_open` is False."""


class RedisRateLimiter:
    def __init__(self, redis: Redis, *, requests_per_window: int, fail_open: bool) -> None:
        self._redis = redis
        self._limit = requests_per_window
        self._fail_open = fail_open

    async def check(self, user_id: str) -> RateLimitDecision:
        now = time.time()
        window = int(now) // _WINDOW_SECONDS
        key = f"rag:ratelimit:user:{user_id}:{window}"
        try:
            count = await self._redis.incr(key)
            if count == 1:  # first request in this window: set the window's own expiry once
                await self._redis.expire(key, _WINDOW_SECONDS)
        except RedisError:
            _LOG.warning(
                "rate limiter: Redis unavailable, %s",
                "failing open" if self._fail_open else "failing closed",
            )
            if self._fail_open:
                return RateLimitDecision(allowed=True)
            raise RateLimitUnavailable("rate limiter backend unavailable") from None

        if count <= self._limit:
            _LOG.info(
                "rate limit decision",
                extra={"user_id": user_id, "count": count, "limit": self._limit, "allowed": True},
            )
            return RateLimitDecision(allowed=True)
        retry_after = _WINDOW_SECONDS - (int(now) % _WINDOW_SECONDS)
        _LOG.info(
            "rate limit decision",
            extra={
                "user_id": user_id, "count": count, "limit": self._limit, "allowed": False,
                "retry_after_seconds": retry_after,
            },
        )
        return RateLimitDecision(allowed=False, retry_after_seconds=float(retry_after))
