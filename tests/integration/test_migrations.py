"""Run against the compose PostgreSQL:

    docker compose up -d postgres
    DATABASE_URL=postgresql+psycopg://...@127.0.0.1:5434/... pytest -q tests/integration
"""
import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("DATABASE_URL")


@pytest.fixture
def engine():  # type: ignore[no-untyped-def]
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    engine = create_engine(DATABASE_URL)
    yield engine
    engine.dispose()


def _has_vector(engine) -> bool:  # type: ignore[no-untyped-def]
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'")).scalar() == 1


def test_baseline_upgrade_downgrade_roundtrip(engine) -> None:  # type: ignore[no-untyped-def]
    cfg = Config("alembic.ini")
    command.downgrade(cfg, "base")
    assert not _has_vector(engine)

    command.upgrade(cfg, "head")
    assert _has_vector(engine)

    command.upgrade(cfg, "head")  # idempotent re-run
    assert _has_vector(engine)
