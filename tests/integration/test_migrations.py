"""Migration round-trip on a THROWAWAY database.

This test runs `alembic downgrade base`, which drops every table. It therefore never touches the
database DATABASE_URL points at: it creates its own `chatbot_test_<hex>` database (like every other
integration test), and the destructive run is additionally guarded by `alembic_env(...,
destructive=True)`, which refuses any database whose name is not on the throwaway convention.

    docker compose up -d postgres
    DATABASE_URL=postgresql+psycopg://...@127.0.0.1:5434/... pytest -q tests/integration
"""
import os

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

from tests.db_safety import UnsafeDatabaseError, alembic_env

pytestmark = pytest.mark.integration

HEAD = ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()


def _has_vector(url: str) -> bool:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname='vector'")).scalar() == 1
    finally:
        engine.dispose()


def _revision(url: str) -> str | None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()  # type: ignore[no-any-return]
    except Exception:
        return None  # no alembic_version table: fully downgraded
    finally:
        engine.dispose()


def test_baseline_upgrade_downgrade_roundtrip(migration_db_url: str) -> None:
    cfg = Config("alembic.ini")
    with alembic_env(migration_db_url, destructive=True):
        command.upgrade(cfg, "head")
        assert _has_vector(migration_db_url) and _revision(migration_db_url) == HEAD

        command.downgrade(cfg, "base")
        assert not _has_vector(migration_db_url)

        command.upgrade(cfg, "head")
        assert _has_vector(migration_db_url) and _revision(migration_db_url) == HEAD

        command.upgrade(cfg, "head")  # idempotent re-run
        assert _revision(migration_db_url) == HEAD


def test_the_roundtrip_leaves_the_configured_database_url_alone(migration_db_url: str) -> None:
    before = os.environ.get("DATABASE_URL")
    with alembic_env(migration_db_url, destructive=True):
        assert os.environ["DATABASE_URL"] == migration_db_url
    assert os.environ.get("DATABASE_URL") == before  # the developer's real URL is untouched


def test_downgrading_a_real_looking_database_is_refused_without_connecting() -> None:
    real = "postgresql+psycopg://chatbot:pw@127.0.0.1:1/chatbot"  # port 1: a connect would fail
    with pytest.raises(UnsafeDatabaseError):
        with alembic_env(real, destructive=True):
            command.downgrade(Config("alembic.ini"), "base")
