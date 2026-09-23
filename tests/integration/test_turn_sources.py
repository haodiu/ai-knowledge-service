"""turn_sources: immutable citation snapshots (Plan §5.4, §12.7; CLAUDE.md invariant #6).

The table must have NO foreign key to documents/document_versions/chunks: cleanup of old
versions must never cascade into citation history.
"""
import uuid

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db import repositories
from app.retrieval.schemas import SourceSnapshot
from tests.integration.conftest import (
    make_chunk,
    make_document,
    make_version,
    point_active,
    unit_vector,
)

pytestmark = pytest.mark.integration


def _snap(**over: object) -> SourceSnapshot:
    base: dict[str, object] = {
        "source_id": uuid.uuid4(), "document_id": uuid.uuid4(),
        "document_version_id": uuid.uuid4(), "chunk_id": uuid.uuid4(),
        "document_title": "Refund policy", "version_no": 3,
        "text_snapshot": "Refunds within 14 days.", "metadata_snapshot": {"chunk_index": 0},
    }
    base.update(over)
    return SourceSnapshot(**base)  # type: ignore[arg-type]


async def _turn(engine: AsyncEngine) -> uuid.UUID:
    created = await repositories.create_turn(engine, user_id="u", question="q")
    return created.turn_id


def test_turn_sources_has_no_foreign_key_to_documents_versions_or_chunks(db_engine: Engine) -> None:
    """Do NOT 'fix' this by adding a FK: cleanup would cascade-delete citation history."""
    with db_engine.connect() as conn:
        referenced = set(conn.execute(text(
            "SELECT confrelid::regclass::text FROM pg_constraint "
            "WHERE conrelid = 'turn_sources'::regclass AND contype = 'f'")).scalars())
        cols = {r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'turn_sources'"))}
    assert referenced == {"turns"}
    assert {"source_id", "turn_id", "document_id", "document_version_id", "chunk_id",
            "document_title", "version_no", "text_snapshot", "metadata_snapshot"} <= cols


def test_source_id_and_turn_version_chunk_are_unique(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        kinds = set(conn.execute(text(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = 'turn_sources'::regclass AND contype = 'u'")).scalars())
    assert any("(source_id)" in k for k in kinds)
    assert any("(turn_id, document_version_id, chunk_id)" in k for k in kinds)


async def test_snapshots_are_saved_with_the_turn_and_can_be_opened_by_source_id(
    async_engine: AsyncEngine,
) -> None:
    turn = await _turn(async_engine)
    snap = _snap()
    await repositories.finish_turn(
        async_engine, turn, graph_status="answered", answer="a", sources=[],
        retrieval_attempts=1, latency_ms=5, snapshots=[snap])
    opened = await repositories.get_turn_source(
        async_engine, turn, snap.source_id, user_id="u"
    )
    assert opened == snap


async def test_a_source_of_another_turn_cannot_be_opened(async_engine: AsyncEngine) -> None:
    turn, other = await _turn(async_engine), await _turn(async_engine)
    snap = _snap()
    await repositories.finish_turn(
        async_engine, turn, graph_status="answered", answer="a", sources=[],
        retrieval_attempts=1, latency_ms=5, snapshots=[snap])
    opened = await repositories.get_turn_source(
        async_engine, other, snap.source_id, user_id="u"
    )
    assert opened is None


async def test_a_different_users_turn_cannot_be_opened(async_engine: AsyncEngine) -> None:
    """Week 6: ownership is checked by user_id, not just by turn_id. Same failure shape as a
    genuinely missing source (invariant #4) -- a 403 here would confirm the turn exists."""
    owner_created = await repositories.create_turn(async_engine, user_id="owner", question="q")
    snap = _snap()
    await repositories.finish_turn(
        async_engine, owner_created.turn_id, graph_status="answered", answer="a", sources=[],
        retrieval_attempts=1, latency_ms=5, snapshots=[snap])

    as_owner = await repositories.get_turn_source(
        async_engine, owner_created.turn_id, snap.source_id, user_id="owner"
    )
    as_intruder = await repositories.get_turn_source(
        async_engine, owner_created.turn_id, snap.source_id, user_id="someone-else"
    )
    assert as_owner == snap
    assert as_intruder is None


async def test_snapshots_survive_deleting_the_document_and_all_its_versions(
    async_engine: AsyncEngine, db_engine: Engine
) -> None:
    with db_engine.begin() as conn:
        doc = make_document(conn, "policy")
        version = make_version(conn, doc, 1, "active")
        chunk = make_chunk(conn, version, 0, "Refunds within 14 days.", unit_vector(0))
        point_active(conn, doc, version)
    turn = await _turn(async_engine)
    snap = _snap(document_id=doc, document_version_id=version, chunk_id=chunk)
    await repositories.finish_turn(
        async_engine, turn, graph_status="answered", answer="a", sources=[],
        retrieval_attempts=1, latency_ms=5, snapshots=[snap])

    with db_engine.begin() as conn:  # what version cleanup does: cascades to versions + chunks
        conn.execute(text("DELETE FROM documents WHERE id = :d"), {"d": doc})
        assert conn.execute(text("SELECT count(*) FROM chunks")).scalar_one() == 0

    opened = await repositories.get_turn_source(
        async_engine, turn, snap.source_id, user_id="u"
    )
    assert opened == snap


async def test_snapshot_keeps_the_old_version_text_after_a_newer_version_activates(
    async_engine: AsyncEngine, db_engine: Engine
) -> None:
    with db_engine.begin() as conn:
        doc = make_document(conn, "policy")
        v1 = make_version(conn, doc, 1, "active")
        c1 = make_chunk(conn, v1, 0, "Refunds within 14 days.", unit_vector(0))
        point_active(conn, doc, v1)
    turn = await _turn(async_engine)
    snap = _snap(document_id=doc, document_version_id=v1, chunk_id=c1, version_no=1,
                 text_snapshot="Refunds within 14 days.")
    await repositories.finish_turn(
        async_engine, turn, graph_status="answered", answer="a", sources=[],
        retrieval_attempts=1, latency_ms=5, snapshots=[snap])

    with db_engine.begin() as conn:  # policy update: v2 supersedes v1
        conn.execute(text("UPDATE document_versions SET status = 'superseded' WHERE id = :v"),
                     {"v": v1})
        v2 = make_version(conn, doc, 2, "active")
        make_chunk(conn, v2, 0, "Refunds within 30 days.", unit_vector(1))
        point_active(conn, doc, v2)

    opened = await repositories.get_turn_source(async_engine, turn, snap.source_id, user_id="u")
    assert opened is not None
    assert opened.version_no == 1 and opened.text_snapshot == "Refunds within 14 days."
    assert opened.document_version_id == v1  # not silently moved to the new active version


async def test_a_failing_snapshot_insert_rolls_the_whole_turn_update_back(
    async_engine: AsyncEngine, db_engine: Engine
) -> None:
    turn = await _turn(async_engine)
    dup = uuid.uuid4()
    with pytest.raises(IntegrityError):
        await repositories.finish_turn(
            async_engine, turn, graph_status="answered", answer="a", sources=[],
            retrieval_attempts=1, latency_ms=5,
            snapshots=[_snap(source_id=dup), _snap(source_id=dup)])
    with db_engine.connect() as conn:
        status, answer = conn.execute(text("SELECT graph_status, answer FROM turns")).one()
        assert conn.execute(text("SELECT count(*) FROM turn_sources")).scalar_one() == 0
    assert (status, answer) == ("running", None)  # no half-finished turn
