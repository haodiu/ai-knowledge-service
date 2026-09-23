"""Week 5: ingestion_jobs queue tracking (Plan §5.3, §12.5, §12.6).

Covers what `test_ingest_idempotency.py`/`test_ingest_edge_cases.py` deliberately don't: the job
row itself -- duplicate delivery, permanent vs transient failure classification, and the stale-
build race (Plan §12.6 rule 7). `ingest_document()`'s own outcomes stay covered by the existing
suites, which pass unchanged (regression gate for this refactor).
"""
import uuid

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

from app.ai.errors import ModelTimeout, ModelUnavailable
from app.db.models import EMBEDDING_DIM
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL, fake_embed
from app.ingestion.service import (
    ActivationError,
    JobNotFoundError,
    StaleVersionError,
    SupersededContentError,
    TransientIngestionError,
    activate_version,
    create_ingestion_job,
    run_ingestion_job,
)
from app.retrieval.schemas import Tier
from tests.integration.conftest import make_chunk, make_document, make_version, point_active

pytestmark = pytest.mark.integration

CONTENT = "Refunds are possible within 30 days.\n\nContact support to start one."


def _create(engine: Engine, *, external_id: str = "docs/refund.md", content: str = CONTENT):  # type: ignore[no-untyped-def]
    return create_ingestion_job(
        engine,
        external_id=external_id,
        title="Refund policy",
        tier=Tier.GENERAL,
        content=content,
        embedding_model=FAKE_EMBEDDING_MODEL,
    )


def _job_row(engine: Engine, job_id: uuid.UUID) -> dict[str, object]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, error_code, retry_count, document_version_id, completed_at "
                "FROM ingestion_jobs WHERE id = :j"
            ),
            {"j": job_id},
        ).one()
    return dict(row._mapping)


def _counts(engine: Engine) -> tuple[int, int]:
    with engine.connect() as conn:
        versions = conn.execute(text("SELECT count(*) FROM document_versions")).scalar_one()
        chunks = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    return versions, chunks


def test_create_ingestion_job_is_idempotent_by_idempotency_key(db_engine: Engine) -> None:
    first = _create(db_engine)
    second = _create(db_engine)  # redelivery of the same create-job request

    assert first.status == second.status == "queued"
    assert first.job_id == second.job_id
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM ingestion_jobs")).scalar_one() == 1


def test_create_ingestion_job_on_already_active_content_creates_no_job(db_engine: Engine) -> None:
    job = _create(db_engine)
    assert job.job_id is not None
    run_ingestion_job(db_engine, job.job_id, embed=fake_embed)

    unchanged = _create(db_engine)  # resubmit identical content now that it's active

    assert unchanged.status == "completed"
    assert unchanged.job_id is None
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM ingestion_jobs")).scalar_one() == 1


def test_run_ingestion_job_twice_does_not_duplicate_chunks_or_reactivate(db_engine: Engine) -> None:
    """Release-blocking (Plan §19): duplicate Celery delivery must not create duplicate chunks or
    the wrong active version."""
    job = _create(db_engine)
    assert job.job_id is not None

    run_ingestion_job(db_engine, job.job_id, embed=fake_embed)
    versions_after_first, chunks_after_first = _counts(db_engine)
    kv_after_first = _knowledge_version(db_engine)

    run_ingestion_job(db_engine, job.job_id, embed=fake_embed)  # redelivery

    assert _counts(db_engine) == (versions_after_first, chunks_after_first)
    assert _knowledge_version(db_engine) == kv_after_first
    assert _job_row(db_engine, job.job_id)["status"] == "completed"


def _knowledge_version(engine: Engine) -> int:
    with engine.connect() as conn:
        kv: int = conn.execute(
            text("SELECT knowledge_version FROM knowledge_base_state")
        ).scalar_one()
    return kv


def test_run_ingestion_job_on_unknown_job_id_raises(db_engine: Engine) -> None:
    with pytest.raises(JobNotFoundError):
        run_ingestion_job(db_engine, uuid.uuid4(), embed=fake_embed)


def test_permanent_embedder_failure_marks_job_failed_with_error_code(db_engine: Engine) -> None:
    job = _create(db_engine)
    assert job.job_id is not None

    def boom(texts: list[str]) -> list[list[float]]:
        raise ModelUnavailable("bad request", retryable=False)

    with pytest.raises(ModelUnavailable):
        run_ingestion_job(db_engine, job.job_id, embed=boom)

    row = _job_row(db_engine, job.job_id)
    assert row["status"] == "failed"
    assert row["error_code"] == "unavailable"


@pytest.mark.parametrize(
    "exc",
    [ModelTimeout("slow"), ModelUnavailable("down", retryable=True)],
    ids=["timeout", "unavailable_retryable"],
)
def test_transient_failure_marks_job_retrying_and_raises_transient_error(
    db_engine: Engine, exc: Exception
) -> None:
    job = _create(db_engine)
    assert job.job_id is not None

    def boom(texts: list[str]) -> list[list[float]]:
        raise exc

    with pytest.raises(TransientIngestionError):
        run_ingestion_job(db_engine, job.job_id, embed=boom)

    row = _job_row(db_engine, job.job_id)
    assert row["status"] == "retrying"
    assert row["retry_count"] == 1


def test_db_operational_error_during_processing_is_transient(db_engine: Engine) -> None:
    job = _create(db_engine)
    assert job.job_id is not None

    def boom(texts: list[str]) -> list[list[float]]:
        raise OperationalError("select 1", {}, ConnectionRefusedError("refused"))

    with pytest.raises(TransientIngestionError):
        run_ingestion_job(db_engine, job.job_id, embed=boom)

    assert _job_row(db_engine, job.job_id)["status"] == "retrying"


def test_retry_after_permanent_failure_recycles_the_job_row(db_engine: Engine) -> None:
    job = _create(db_engine)
    assert job.job_id is not None

    def boom(texts: list[str]) -> list[list[float]]:
        raise ModelUnavailable("bad", retryable=False)

    with pytest.raises(ModelUnavailable):
        run_ingestion_job(db_engine, job.job_id, embed=boom)
    assert _job_row(db_engine, job.job_id)["status"] == "failed"

    retried = _create(db_engine)  # resubmit the exact same content

    assert retried.job_id == job.job_id  # same row, recycled -- not a second job
    assert retried.status == "queued"
    run_ingestion_job(db_engine, retried.job_id, embed=fake_embed)  # type: ignore[arg-type]
    row = _job_row(db_engine, job.job_id)
    assert row["status"] == "completed"
    assert row["error_code"] is None


def test_resubmitting_superseded_content_creates_a_failed_audit_job_without_touching_active(
    db_engine: Engine,
) -> None:
    v1 = _create(db_engine, content="V1 text")
    assert v1.job_id is not None
    run_ingestion_job(db_engine, v1.job_id, embed=fake_embed)
    v2 = _create(db_engine, content="V2 text")
    assert v2.job_id is not None
    run_ingestion_job(db_engine, v2.job_id, embed=fake_embed)

    with pytest.raises(SupersededContentError):
        _create(db_engine, content="V1 text")

    with db_engine.connect() as conn:
        rejected = conn.execute(
            text(
                "SELECT status, error_code, document_version_id FROM ingestion_jobs "
                "WHERE error_code = 'superseded_content'"
            )
        ).one()
        active_version_no = conn.execute(
            text(
                "SELECT dv.version_no FROM documents d "
                "JOIN document_versions dv ON dv.id = d.active_version_id"
            )
        ).scalar_one()
    assert (rejected.status, rejected.document_version_id) == ("failed", None)
    assert active_version_no == 2  # unchanged by the rejected resubmission


def test_stale_build_is_marked_superseded_not_failed(db_engine: Engine) -> None:
    """Plan §12.6 rule 7: two jobs race for the same document; the one that loses (its version is
    no longer newer than what's already active by the time it tries to activate) is `superseded`.
    """
    with db_engine.begin() as conn:
        doc = make_document(conn, "policies/race.md")
        v1 = make_version(conn, doc, 1, "building", content="v1")
        v2 = make_version(conn, doc, 2, "building", content="v2")
        make_chunk(conn, v1, 0, "chunk v1", [0.0] * EMBEDDING_DIM)
        make_chunk(conn, v2, 0, "chunk v2", [0.0] * EMBEDDING_DIM)

    activate_version(db_engine, v2)  # the newer job wins the race first

    with pytest.raises(StaleVersionError):
        activate_version(db_engine, v1)  # the older job then loses


def test_activate_version_with_job_id_completes_job_atomically(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        doc = make_document(conn, "policies/atomic.md")
        v1 = make_version(conn, doc, 1, "active")
        point_active(conn, doc, v1)
        make_chunk(conn, v1, 0, "old", [0.0] * EMBEDDING_DIM)
        v2 = make_version(conn, doc, 2, "building")
        make_chunk(conn, v2, 0, "new", [0.0] * EMBEDDING_DIM)
        job_id = conn.execute(
            text(
                "INSERT INTO ingestion_jobs (document_id, document_version_id, version_no, "
                "content_hash, idempotency_key, status) "
                "VALUES (:d, :v, 2, 'h', 'k', 'processing') RETURNING id"
            ),
            {"d": doc, "v": v2},
        ).scalar_one()

    activate_version(db_engine, v2, job_id=job_id)

    row = _job_row(db_engine, job_id)
    assert row["status"] == "completed"
    assert row["document_version_id"] == v2
    assert row["completed_at"] is not None


def test_activate_version_failure_leaves_job_unfinished(db_engine: Engine) -> None:
    """Invariant 9 extended to job completion: an activation that never happens (version has no
    chunks) must not silently mark the job done either."""
    with db_engine.begin() as conn:
        doc = make_document(conn, "policies/no-chunks.md")
        v1 = make_version(conn, doc, 1, "building")
        job_id = conn.execute(
            text(
                "INSERT INTO ingestion_jobs (document_id, document_version_id, version_no, "
                "content_hash, idempotency_key, status) "
                "VALUES (:d, :v, 1, 'h', 'k', 'processing') RETURNING id"
            ),
            {"d": doc, "v": v1},
        ).scalar_one()

    with pytest.raises(ActivationError):
        activate_version(db_engine, v1, job_id=job_id)

    assert _job_row(db_engine, job_id)["status"] == "processing"  # unchanged, not "completed"
