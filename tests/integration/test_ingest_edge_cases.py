"""Ingestion failure, retry and duplicate-delivery behaviour (invariants 9 and 10, decision E)."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import Engine, text

from app.db.models import EMBEDDING_DIM
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL, fake_embed
from app.ingestion.service import (
    IngestionError,
    IngestResult,
    SupersededContentError,
    ingest_document,
)
from app.retrieval.schemas import Tier

pytestmark = pytest.mark.integration

V1 = "Refunds are possible within 30 days.\n\nContact support to start one."
V2 = "Refunds are possible within 14 days.\n\nContact support to start one."


def _ingest(
    engine: Engine,
    content: str,
    *,
    embed=fake_embed,  # type: ignore[no-untyped-def]
    tier: Tier = Tier.GENERAL,
    model: str = FAKE_EMBEDDING_MODEL,
) -> IngestResult:
    return ingest_document(
        engine,
        external_id="docs/refund.md",
        title="Refund policy",
        tier=tier,
        content=content,
        embed=embed,
        embedding_model=model,
    )


def _statuses(engine: Engine) -> dict[int, str]:
    with engine.connect() as conn:
        return dict(
            conn.execute(
                text("SELECT version_no, status FROM document_versions ORDER BY version_no")
            ).all()
        )


def _scalar(engine: Engine, sql: str) -> object:
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar_one()


def _boom(texts: list[str]) -> list[list[float]]:
    raise RuntimeError("provider down")


def test_embedder_failure_marks_the_new_version_failed_and_old_keeps_serving(
    db_engine: Engine,
) -> None:
    _ingest(db_engine, V1)

    with pytest.raises(RuntimeError, match="provider down"):
        _ingest(db_engine, V2, embed=_boom)

    assert _statuses(db_engine) == {1: "active", 2: "failed"}
    assert _scalar(db_engine, "SELECT knowledge_version FROM knowledge_base_state") == 1
    assert _scalar(
        db_engine,
        "SELECT dv.version_no FROM documents d JOIN document_versions dv "
        "ON dv.id = d.active_version_id",
    ) == 1


def test_wrong_embedding_dimension_is_rejected_and_nothing_activates(db_engine: Engine) -> None:
    _ingest(db_engine, V1)

    with pytest.raises(IngestionError, match=str(EMBEDDING_DIM)):
        _ingest(db_engine, V2, embed=lambda ts: [[0.1, 0.2] for _ in ts])

    assert _statuses(db_engine) == {1: "active", 2: "failed"}


def test_retry_after_failure_rebuilds_the_same_version_row(db_engine: Engine) -> None:
    """Decision E, branch 2: the unique key forbids a second row, so the failed one is rebuilt."""
    _ingest(db_engine, V1)
    with pytest.raises(RuntimeError):
        _ingest(db_engine, V2, embed=_boom)

    result = _ingest(db_engine, V2)

    assert (result.version_no, result.outcome) == (2, "activated")
    assert _statuses(db_engine) == {1: "superseded", 2: "active"}
    assert _scalar(db_engine, "SELECT count(*) FROM document_versions") == 2
    assert _scalar(db_engine, "SELECT knowledge_version FROM knowledge_base_state") == 2


def test_a_stuck_building_version_is_rebuilt_without_leftover_chunks(db_engine: Engine) -> None:
    """A crash can leave `building` plus partial chunks; a re-run must replace, not append."""
    _ingest(db_engine, V1)
    with pytest.raises(RuntimeError):
        _ingest(db_engine, V2, embed=_boom)
    with db_engine.begin() as conn:
        conn.execute(text("UPDATE document_versions SET status = 'building' WHERE version_no = 2"))
        conn.execute(
            text(
                "INSERT INTO chunks (document_version_id, chunk_index, text, embedding) "
                "SELECT id, 99, 'stray partial chunk', "
                "array_fill(0.1::real, ARRAY[:n])::vector FROM document_versions "
                "WHERE version_no = 2"
            ),
            {"n": EMBEDDING_DIM},
        )

    _ingest(db_engine, V2)

    assert _scalar(db_engine, "SELECT count(*) FROM chunks WHERE text = 'stray partial chunk'") == 0
    assert _statuses(db_engine) == {1: "superseded", 2: "active"}


def test_resubmitting_superseded_content_is_refused_and_changes_nothing(
    db_engine: Engine,
) -> None:
    """Decision E, branch 3: A -> B -> A is an error in Week 2, not an implicit rollback."""
    _ingest(db_engine, V1)
    _ingest(db_engine, V2)

    with pytest.raises(SupersededContentError):
        _ingest(db_engine, V1)

    assert _statuses(db_engine) == {1: "superseded", 2: "active"}
    assert _scalar(db_engine, "SELECT knowledge_version FROM knowledge_base_state") == 2


def test_changed_embedding_model_reindexes_identical_content(db_engine: Engine) -> None:
    """Plan §12.6 rule 2: same content, different index config -> a new version."""
    first = _ingest(db_engine, V1, model="model-a")
    second = _ingest(db_engine, V1, model="model-b")

    assert second.version_id != first.version_id
    assert (second.version_no, second.outcome) == (2, "activated")


def test_changing_the_tier_of_an_existing_document_is_refused(db_engine: Engine) -> None:
    """Tier lives on `documents`; applying it early would re-scope the live version at once."""
    _ingest(db_engine, V1, tier=Tier.INTERNAL)

    with pytest.raises(IngestionError, match="tier change"):
        _ingest(db_engine, V2, tier=Tier.GENERAL)

    assert _scalar(db_engine, "SELECT tier FROM documents") == "internal"
    assert _statuses(db_engine) == {1: "active"}


def test_archived_document_does_not_accept_new_versions(db_engine: Engine) -> None:
    _ingest(db_engine, V1)
    with db_engine.begin() as conn:
        conn.execute(text("UPDATE documents SET status = 'archived'"))

    with pytest.raises(IngestionError, match="archived"):
        _ingest(db_engine, V2)

    assert _statuses(db_engine) == {1: "active"}


def test_empty_content_is_rejected_before_touching_the_database(db_engine: Engine) -> None:
    with pytest.raises(IngestionError):
        _ingest(db_engine, "  \n\n ")
    assert _scalar(db_engine, "SELECT count(*) FROM documents") == 0


def test_concurrent_duplicate_delivery_creates_one_version_and_no_duplicate_chunks(
    db_engine: Engine,
) -> None:
    """At-least-once delivery: the same job running several times at once (invariant 10)."""
    workers = 6
    barrier = threading.Barrier(workers)

    def run(_: int) -> IngestResult:
        barrier.wait()
        return _ingest(db_engine, V1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, range(workers)))

    assert sorted(r.outcome for r in results) == ["activated"] + ["unchanged"] * (workers - 1)
    assert len({r.version_id for r in results}) == 1
    assert _statuses(db_engine) == {1: "active"}
    assert _scalar(db_engine, "SELECT knowledge_version FROM knowledge_base_state") == 1
    assert _scalar(
        db_engine, "SELECT count(*) FROM chunks"
    ) == _scalar(
        db_engine, "SELECT count(DISTINCT chunk_index) FROM chunks"
    )
