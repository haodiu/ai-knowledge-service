from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app.api import health
from app.db.session import create_engine
from app.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Engine and Redis client connect lazily, so startup never blocks on a dependency.
        timeout = settings.health_check_timeout_seconds
        app.state.settings = settings
        app.state.engine = create_engine(settings)
        app.state.redis = Redis.from_url(
            settings.redis_url.get_secret_value(),
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )
        try:
            yield
        finally:
            await app.state.redis.aclose()
            await app.state.engine.dispose()

    app = FastAPI(title="RAG Chatbot Service", lifespan=lifespan)
    app.include_router(health.router)
    return app
