"""Invariant #10: ingestion is idempotent by design. Plan §12.6 rules 1, 2, 5.

Only behaviour the Plan spells out is tested here. Re-submitting content identical to a
*superseded* or *failed* version (decision E) is intentionally NOT covered until it is approved.
"""
import pytest
from sqlalchemy import Engine, text

from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL, fake_embed
from app.ingestion.service import ingest_document
from app.retrieval.schemas import Tier

pytestmark = pytest.mark.integration

V1 = "# Refund policy\n\nRefunds are possible within 30 days.\n\nContact support to start one."
V2 = "# Refund policy\n\nRefunds are possible within 14 days.\n\nContact support to start one."


def _ingest(engine: Engine, content: str):  # type: ignore[no-untyped-def]
    return ingest_document(
        engine,
        external_id="docs/refund.md",
        title="Refund policy",
        tier=Tier.GENERAL,
        content=content,
        embed=fake_embed,
        embedding_model=FAKE_EMBEDDING_MODEL,
    )


def _counts(engine: Engine) -> tuple[int, int, int]:
    with engine.connect() as conn:
        versions = conn.execute(text("SELECT count(*) FROM document_versions")).scalar_one()
        chunks = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        kv = conn.execute(text("SELECT knowledge_version FROM knowledge_base_state")).scalar_one()
    return versions, chunks, kv


def test_first_ingest_activates_version_1(db_engine: Engine) -> None:
    result = _ingest(db_engine, V1)

    assert (result.version_no, result.outcome) == (1, "activated")
    versions, chunks, kv = _counts(db_engine)
    assert (versions, kv) == (1, 1)
    assert chunks >= 1
    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT dv.status, d.active_version_id = dv.id AS pointed "
                "FROM documents d JOIN document_versions dv ON dv.document_id = d.id"
            )
        ).one()
    assert (row.status, row.pointed) == ("active", True)


def test_ingesting_identical_content_twice_changes_nothing(db_engine: Engine) -> None:
    first = _ingest(db_engine, V1)
    after_first = _counts(db_engine)

    second = _ingest(db_engine, V1)  # redelivery / re-run

    assert second.outcome == "unchanged"
    assert second.version_id == first.version_id
    assert _counts(db_engine) == after_first  # no new version, no duplicate chunks, no kv bump


def test_changed_content_creates_version_2_and_supersedes_version_1(db_engine: Engine) -> None:
    first = _ingest(db_engine, V1)
    second = _ingest(db_engine, V2)

    assert (second.version_no, second.outcome) == (2, "activated")
    assert second.document_id == first.document_id
    with db_engine.connect() as conn:
        statuses = dict(
            conn.execute(
                text("SELECT version_no, status FROM document_versions ORDER BY version_no")
            ).all()
        )
        kv = conn.execute(text("SELECT knowledge_version FROM knowledge_base_state")).scalar_one()
    assert statuses == {1: "superseded", 2: "active"}
    assert kv == 2
