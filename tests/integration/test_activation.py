"""Invariant #9 (release-blocking): version activation is one transaction.

Written BEFORE the implementation. Failure is injected with test-only triggers so the rollback is
exercised in the real database, not mocked.
"""
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from app.ingestion.service import ActivationError, activate_version
from tests.integration.conftest import (
    make_chunk,
    make_document,
    make_version,
    point_active,
    unit_vector,
)

pytestmark = pytest.mark.integration


@dataclass
class Setup:
    doc: uuid.UUID
    old: uuid.UUID  # version 1, active, serving traffic
    new: uuid.UUID  # version 2, building, fully written


@pytest.fixture
def seeded(db_engine: Engine) -> Setup:
    with db_engine.begin() as conn:
        doc = make_document(conn, "activation")
        old = make_version(conn, doc, 1, "active")
        point_active(conn, doc, old)
        make_chunk(conn, old, 0, "old policy text", unit_vector(0))
        new = make_version(conn, doc, 2, "building")
        make_chunk(conn, new, 0, "new policy text", unit_vector(1))
        make_chunk(conn, new, 1, "new policy more", unit_vector(2))
    return Setup(doc, old, new)


def _state(engine: Engine, s: Setup) -> dict[str, object]:
    with engine.connect() as conn:
        statuses = dict(
            conn.execute(
                text("SELECT version_no, status FROM document_versions WHERE document_id = :d"),
                {"d": s.doc},
            ).all()
        )
        pointer = conn.execute(
            text("SELECT active_version_id FROM documents WHERE id = :d"), {"d": s.doc}
        ).scalar_one()
        kv = conn.execute(text("SELECT knowledge_version FROM knowledge_base_state")).scalar_one()
        chunks = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        activated = dict(
            conn.execute(
                text("SELECT version_no, activated_at IS NOT NULL FROM document_versions "
                     "WHERE document_id = :d"),
                {"d": s.doc},
            ).all()
        )
    return {
        "statuses": statuses,
        "pointer": pointer,
        "knowledge_version": kv,
        "chunks": chunks,
        "activated": activated,
    }


def test_activation_switches_everything_together(db_engine: Engine, seeded: Setup) -> None:
    activate_version(db_engine, seeded.new)

    state = _state(db_engine, seeded)
    assert state["statuses"] == {1: "superseded", 2: "active"}
    assert state["pointer"] == seeded.new
    assert state["knowledge_version"] == 1
    assert state["activated"][2] is True  # type: ignore[index]
    assert state["chunks"] == 3  # nothing deleted: the old version stays for rollback (§12.7)


# Each trigger fires on a different statement of the activation, so whichever order the
# implementation uses, at least one earlier statement has already run when the failure hits.
FAILURE_POINTS = {
    "knowledge_version bump": ("knowledge_base_state", "BEFORE UPDATE", ""),
    "pointer update": ("documents", "BEFORE UPDATE", ""),
    "promote new version": ("document_versions", "BEFORE UPDATE", "WHEN (NEW.status = 'active')"),
    "supersede old version": (
        "document_versions",
        "BEFORE UPDATE",
        "WHEN (NEW.status = 'superseded')",
    ),
}


@pytest.fixture
def failure_trigger(db_engine: Engine) -> Iterator[object]:
    installed: list[tuple[str, str]] = []

    def install(table: str, timing: str, when: str) -> None:
        name = f"inject_failure_{len(installed)}"
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS "
                    "$$ BEGIN RAISE EXCEPTION 'injected activation failure'; END $$"
                )
            )
            conn.execute(
                text(f"CREATE TRIGGER {name} {timing} ON {table} FOR EACH ROW {when} "
                     f"EXECUTE FUNCTION {name}()")
            )
        installed.append((name, table))

    yield install
    with db_engine.begin() as conn:
        for name, table in installed:
            conn.execute(text(f"DROP TRIGGER IF EXISTS {name} ON {table}"))
            conn.execute(text(f"DROP FUNCTION IF EXISTS {name}()"))


@pytest.mark.parametrize("point", list(FAILURE_POINTS))
def test_failure_mid_activation_leaves_the_old_version_serving(
    db_engine: Engine, seeded: Setup, failure_trigger: object, point: str
) -> None:
    before = _state(db_engine, seeded)
    assert before["statuses"] == {1: "active", 2: "building"}

    failure_trigger(*FAILURE_POINTS[point])  # type: ignore[operator]
    with pytest.raises(DBAPIError, match="injected activation failure"):
        activate_version(db_engine, seeded.new)

    after = _state(db_engine, seeded)
    # status, pointer, knowledge_version, chunks and activated_at: all unchanged
    assert after == before
    assert after["pointer"] == seeded.old


@pytest.mark.parametrize("status", ["active", "superseded", "failed"])
def test_only_a_building_version_can_be_activated(
    db_engine: Engine, seeded: Setup, status: str
) -> None:
    with db_engine.begin() as conn:
        if status == "active":
            # Make v1 the "new" one being re-activated: it is already active.
            target = seeded.old
        else:
            conn.execute(
                text("UPDATE document_versions SET status = :s WHERE id = :v"),
                {"s": status, "v": seeded.new},
            )
            target = seeded.new
    before = _state(db_engine, seeded)

    with pytest.raises(ActivationError):
        activate_version(db_engine, target)

    assert _state(db_engine, seeded) == before


def test_a_version_without_chunks_is_never_activated(db_engine: Engine, seeded: Setup) -> None:
    """Rule 4 (Plan §12.6): activate only after all chunks are written. An empty version would
    silently take the old, working version out of service."""
    with db_engine.begin() as conn:
        empty = make_version(conn, seeded.doc, 3, "building")
    before = _state(db_engine, seeded)

    with pytest.raises(ActivationError):
        activate_version(db_engine, empty)

    assert _state(db_engine, seeded) == before


def test_unknown_version_id_is_rejected(db_engine: Engine, seeded: Setup) -> None:
    before = _state(db_engine, seeded)
    with pytest.raises(ActivationError):
        activate_version(db_engine, uuid.uuid4())
    assert _state(db_engine, seeded) == before


def test_a_stale_build_cannot_roll_back_the_active_version(db_engine: Engine) -> None:
    """version_no 1 finishing after version_no 2 went active must not replace it (Plan §12.6/7)."""
    with db_engine.begin() as conn:
        doc = make_document(conn, "stale")
        stale = make_version(conn, doc, 1, "building")
        make_chunk(conn, stale, 0, "slow old build", unit_vector(0))
        current = make_version(conn, doc, 2, "active")
        point_active(conn, doc, current)
        make_chunk(conn, current, 0, "current", unit_vector(1))
    s = Setup(doc, old=current, new=stale)
    before = _state(db_engine, s)

    with pytest.raises(ActivationError):
        activate_version(db_engine, stale)

    assert _state(db_engine, s) == before
    assert before["pointer"] == current
