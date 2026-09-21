"""Integration fixtures: one throwaway database per test session, migrated with Alembic.

Needs DATABASE_URL pointing at a server where the user may CREATE DATABASE (the compose
postgres user is a superuser). The database named in DATABASE_URL itself is never touched,
so these tests cannot clobber dev data and do not race test_migrations.py.
"""
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.db.models import EMBEDDING_DIM
from app.settings import get_settings

DATABASE_URL = os.environ.get("DATABASE_URL")


def _render(url: Any) -> str:
    return url.render_as_string(hide_password=False)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def test_db_url() -> Iterator[str]:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    base = make_url(DATABASE_URL)
    name = f"chatbot_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(_render(base.set(database="postgres")), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    test_url = _render(base.set(database=name))

    try:
        _migrate(test_url)
        yield test_url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _migrate(url: str) -> None:
    """Run `alembic upgrade head` against `url`, then put the environment back exactly as found.

    migrations/env.py resolves the URL through get_settings(), so the override is unavoidable, but
    it must NOT outlive this call: test_migrations.py also drives Alembic through env.py and would
    otherwise migrate (and downgrade) this temp database instead of the one it targets.
    """
    keys = ("DATABASE_URL", "REDIS_URL", "RABBITMQ_URL")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["DATABASE_URL"] = url
    os.environ.setdefault("REDIS_URL", "redis://localhost:1/0")
    os.environ.setdefault("RABBITMQ_URL", "amqp://u:p@localhost:1//")
    get_settings.cache_clear()
    try:
        command.upgrade(Config("alembic.ini"), "head")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@pytest.fixture
def db_engine(test_db_url: str) -> Iterator[Engine]:
    """Sync engine on the migrated temp DB, reset to the just-migrated state for every test."""
    engine = create_engine(test_db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE documents, document_versions, chunks, conversations, turns, "
                "model_calls, turn_sources"
            )
        )
        conn.execute(text("UPDATE knowledge_base_state SET knowledge_version = 0"))
    yield engine
    engine.dispose()


@pytest.fixture
async def async_engine(test_db_url: str, db_engine: Engine) -> AsyncIterator[AsyncEngine]:
    """Async engine (retrieval is async); depends on db_engine so the tables are reset first."""
    engine = create_async_engine(test_db_url)
    yield engine
    await engine.dispose()


# --- seeding helpers -----------------------------------------------------------------------
# Tests build DB states directly with SQL instead of going through ingestion, so that
# retrieval/activation tests do not depend on the code they are supposed to constrain.


def unit_vector(axis: int, *, blend: float = 0.0, other_axis: int = 1) -> list[float]:
    """Basis vector e_axis, optionally mixed with e_other_axis. Cosine distance is controllable."""
    vec = [0.0] * EMBEDDING_DIM
    vec[axis] = 1.0
    if blend:
        vec[other_axis] = blend
    return vec


def vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in vec) + "]"


def make_document(
    conn: Connection, external_id: str, *, tier: str = "general", status: str = "active"
) -> uuid.UUID:
    return conn.execute(
        text(
            "INSERT INTO documents (external_id, title, tier, status) "
            "VALUES (:e, :t, :tier, :status) RETURNING id"
        ),
        {"e": external_id, "t": f"Title {external_id}", "tier": tier, "status": status},
    ).scalar_one()


def make_version(
    conn: Connection,
    document_id: uuid.UUID,
    version_no: int,
    status: str,
    *,
    content: str = "content",
) -> uuid.UUID:
    return conn.execute(
        text(
            "INSERT INTO document_versions (document_id, version_no, content, content_hash, "
            "chunking_version, embedding_model, index_config_hash, status, activated_at) "
            "VALUES (:d, :n, :c, :h, 'test-chunking', 'test-model', 'test-cfg', :s, "
            "CASE WHEN :s = 'active' THEN now() END) RETURNING id"
        ),
        {"d": document_id, "n": version_no, "c": content, "h": f"hash-{version_no}", "s": status},
    ).scalar_one()


def make_chunk(
    conn: Connection, version_id: uuid.UUID, index: int, body: str, embedding: list[float]
) -> uuid.UUID:
    return conn.execute(
        text(
            "INSERT INTO chunks (document_version_id, chunk_index, text, embedding) "
            "VALUES (:v, :i, :t, CAST(:e AS vector)) RETURNING id"
        ),
        {"v": version_id, "i": index, "t": body, "e": vec_literal(embedding)},
    ).scalar_one()


def point_active(conn: Connection, document_id: uuid.UUID, version_id: uuid.UUID) -> None:
    conn.execute(
        text("UPDATE documents SET active_version_id = :v WHERE id = :d"),
        {"v": version_id, "d": document_id},
    )
