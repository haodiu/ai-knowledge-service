from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app.api import conversations, health, ingestions
from app.api.middleware import RequestIdMiddleware
from app.db.session import create_engine, create_sync_engine
from app.logging_setup import configure_logging
from app.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # Before anything else -- including for a test session that only ever builds the app and never
    # enters a real request (so a route-less "does create_app() blow up" test still gets JSON logs).
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Engine and Redis client connect lazily, so startup never blocks on a dependency.
        # app.state.models/prompts are NOT built here: they are lazy, cached on first use, in
        # app.api.dependencies (get_models/get_prompts) -- building the real ModelRegistry needs
        # a live GEMINI_API_KEY, and a route that never touches models must never fail app
        # startup over a missing one.
        timeout = settings.health_check_timeout_seconds
        app.state.settings = settings
        app.state.engine = create_engine(settings)
        app.state.sync_engine = create_sync_engine(settings)  # app.ingestion.service is sync
        app.state.redis = Redis.from_url(
            settings.redis_url.get_secret_value(),
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )
        try:
            yield
        finally:
            await app.state.redis.aclose()
            app.state.sync_engine.dispose()
            await app.state.engine.dispose()

    app = FastAPI(title="RAG Chatbot Service", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    app.include_router(conversations.router)
    app.include_router(ingestions.router)
    return app
