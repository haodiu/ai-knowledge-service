import asyncio
import secrets
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from kombu import Connection
from redis.asyncio import Redis
from sqlalchemy import Engine, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.errors import ConfigurationError
from app.ai.prompts.loader import Prompts, load_prompts
from app.ai.registry import ModelRegistry, build_registry
from app.auth.jwt import InvalidTokenError, verify_token
from app.auth.policies import AuthorizationContext
from app.rate_limit.policy import RateLimiter
from app.rate_limit.redis import RateLimitUnavailable, RedisRateLimiter
from app.settings import Settings
from app.tools.subscription import SubscriptionToolClient, build_tool_client

# A probe raises on failure and returns None on success.
Probe = Callable[[], Awaitable[None]]


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.engine
    return engine


def get_sync_engine(request: Request) -> Engine:
    """Sync engine for routes calling the (sync) ingestion service (app.ingestion.service)."""
    engine: Engine = request.app.state.sync_engine
    return engine


def get_redis(request: Request) -> Redis:
    redis: Redis = request.app.state.redis
    return redis


def get_models(
    request: Request, settings: Annotated[Settings, Depends(get_settings_dep)]
) -> ModelRegistry:
    """Built once, lazily, and cached on app.state -- not per request (Plan §11.4 routing lives in
    settings; building the real adapters needs a live GEMINI_API_KEY, so this stays lazy rather
    than running in the app lifespan: a route that never touches models -- /healthz, /readyz --
    must never fail app startup over a missing key)."""
    if not hasattr(request.app.state, "models"):
        request.app.state.models = build_registry(settings)
    models: ModelRegistry = request.app.state.models
    return models


def get_tool_client(
    request: Request, settings: Annotated[Settings, Depends(get_settings_dep)]
) -> SubscriptionToolClient | None:
    """Built once, lazily, and cached on app.state (Plan §10, §18 Tuan 7).

    Unlike `get_models` -- every turn needs a chat model, so failing the whole request when one
    can't be built is correct there -- NOT every turn needs the subscription tool (only a plan
    that proposes one does), and no Payment/Subscription host exists yet. So this never raises:
    a missing SUBSCRIPTION_SERVICE_BASE_URL/TOKEN yields `None` here, and only turns whose plan
    actually calls the tool see a (turn-scoped, `temporarily_unavailable`) consequence, in
    `app/graph/nodes/tools.py`. Raising ConfigurationError here instead would fail EVERY request,
    including ones that never touch the tool at all -- FastAPI resolves this dependency before the
    route handler runs, so that failure would not even be a graceful HTTP error."""
    if not hasattr(request.app.state, "tool_client"):
        try:
            request.app.state.tool_client = build_tool_client(settings)
        except ConfigurationError:
            request.app.state.tool_client = None
    tool_client: SubscriptionToolClient | None = request.app.state.tool_client
    return tool_client


def get_prompts(
    request: Request, settings: Annotated[Settings, Depends(get_settings_dep)]
) -> Prompts:
    if not hasattr(request.app.state, "prompts"):
        request.app.state.prompts = load_prompts(settings.prompt_version)
    prompts: Prompts = request.app.state.prompts
    return prompts


def get_auth(
    request: Request, settings: Annotated[Settings, Depends(get_settings_dep)]
) -> AuthorizationContext:
    """Verify the host-minted JWT (Plan §7). The only place a request's `Authorization` header is
    read; everything downstream only ever sees the resulting AuthorizationContext."""
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    try:
        return verify_token(token, settings)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid token") from exc


def get_service_auth(
    request: Request, settings: Annotated[Settings, Depends(get_settings_dep)]
) -> None:
    """Separate trust boundary from get_auth: host-to-host for the internal ingestion endpoint,
    no tier/AuthorizationContext involved (Plan §6)."""
    provided = request.headers.get("X-Service-Token", "")
    expected = settings.ingestion_service_token.get_secret_value()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid service token")


def get_rate_limiter(settings: Annotated[Settings, Depends(get_settings_dep)],
                     redis: Annotated[Redis, Depends(get_redis)]) -> RateLimiter:
    return RedisRateLimiter(
        redis,
        requests_per_window=settings.rate_limit_requests_per_minute,
        fail_open=settings.rate_limit_fail_open,
    )


async def enforce_rate_limit(
    auth: Annotated[AuthorizationContext, Depends(get_auth)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> None:
    try:
        decision = await limiter.check(auth.user_id)
    except RateLimitUnavailable as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limiter unavailable"
        ) from exc
    if not decision.allowed:
        headers = (
            {"Retry-After": str(int(decision.retry_after_seconds))}
            if decision.retry_after_seconds is not None
            else None
        )
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded",
                            headers=headers)


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
