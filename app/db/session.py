from sqlalchemy import Engine
from sqlalchemy import create_engine as sa_create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.settings import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
    )


def create_sync_engine(settings: Settings) -> Engine:
    """Sync engine for the ingestion service/CLI (the online retrieval path is async)."""
    return sa_create_engine(settings.database_url.get_secret_value(), pool_pre_ping=True)
