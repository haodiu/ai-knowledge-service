import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated

from fastapi import Depends, Request
from kombu import Connection
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.settings import Settings

# A probe raises on failure and returns None on success.
Probe = Callable[[], Awaitable[None]]


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.engine
    return engine


def get_redis(request: Request) -> Redis:
    redis: Redis = request.app.state.redis
    return redis


def get_probes(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    engine: Annotated[AsyncEngine, Depends(get_engine)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> Mapping[str, Probe]:
    """Readiness probes keyed by dependency name. Overridable in tests."""

    async def postgres() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def redis_ping() -> None:
        await redis.ping()

    def _amqp_connect() -> None:
        # kombu is blocking; run in a thread. No retries: a probe must answer fast.
        timeout = settings.health_check_timeout_seconds
        with Connection(settings.rabbitmq_url.get_secret_value(), connect_timeout=timeout) as conn:
            conn.ensure_connection(max_retries=0, timeout=timeout)

    async def rabbitmq() -> None:
        await asyncio.to_thread(_amqp_connect)

    return {"postgres": postgres, "rabbitmq": rabbitmq, "redis": redis_ping}
