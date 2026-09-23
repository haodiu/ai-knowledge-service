"""Week 5 CLI subcommands: `ingest --queue`, `retry-failed`, `republish-stale`,
`cleanup-versions` (Plan §12.3, §12.7, §12.8).

None of these need a live broker: `app.ingestion.cli._dispatch` is stubbed to run the job inline
via the real `run_ingestion_job()` with the fake embedder, instead of actually publishing to
Celery/RabbitMQ. That exercises the exact same job-selection/status-reset SQL the real `_dispatch`
runs, and the exact same processing function the real worker calls -- only the transport differs
(CLAUDE.md: "don't require a live broker for the fast test loop").
"""
import uuid
from pathlib import Path

import pytest
from sqlalchemy import Connection, Engine, text

from app.ai.errors import ModelUnavailable
from app.ingestion.cli import main
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL, fake_embed
from app.ingestion.service import (
    SupersededContentError,
    create_ingestion_job,
    run_ingestion_job,
)
from app.retrieval.schemas import Tier
from app.settings import get_settings
from tests.integration.conftest import make_chunk, make_document, make_version, point_active

pytestmark = pytest.mark.integration

GENERAL = "---\ntitle: Refund policy\ntier: general\n---\nRefunds within 14 days.\n"


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, test_db_url: str, db_engine: Engine) -> Engine:
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:1/0")
    monkeypatch.setenv("RABBITMQ_URL", "amqp://u:p@localhost:1//")
    get_settings.cache_clear()
    yield db_engine  # type: ignore[misc]
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _inline_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real `.delay()`-publishing `_dispatch` with one that runs the job right here,
    with the deterministic fake embedder -- no broker, no real API key, no Celery singleton."""

    def run_inline(engine: Engine, job_id: uuid.UUID) -> None:
        run_ingestion_job(engine, job_id, embed=fake_embed)

    monkeypatch.setattr("app.ingestion.cli._dispatch", run_inline)


def _status(engine: Engine, job_id: uuid.UUID) -> str:
    with engine.connect() as conn:
        status: str = conn.execute(
            text("SELECT status FROM ingestion_jobs WHERE id = :j"), {"j": job_id}
        ).scalar_one()
    return status


def test_ingest_queue_flag_creates_and_completes_a_job(
    cli_env: Engine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-used-by-the-inline-stub")
    get_settings.cache_clear()
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "refund.md").write_text(GENERAL, encoding="utf-8")

    assert main(["ingest", str(tmp_path), "--queue"]) == 0

    out = capsys.readouterr().out
    assert "queued (job" in out
    with cli_env.connect() as conn:
        status = conn.execute(text("SELECT status FROM ingestion_jobs")).scalar_one()
        active_status = conn.execute(
            text("SELECT status FROM document_versions WHERE version_no = 1")
        ).scalar_one()
    assert status == "completed"
    assert active_status == "active"


def test_retry_failed_requeues_retryable_jobs_and_skips_superseded_content(
    cli_env: Engine,
) -> None:
    # A: a genuinely failed job (permanent embedder error) -- eligible for retry.
    job_a = create_ingestion_job(
        cli_env,
        external_id="a.md",
        title="A",
        tier=Tier.GENERAL,
        content="A content",
        embedding_model=FAKE_EMBEDDING_MODEL,
    )
    assert job_a.job_id is not None
    with pytest.raises(ModelUnavailable):
        run_ingestion_job(
            cli_env,
            job_a.job_id,
            embed=lambda texts: (_ for _ in ()).throw(ModelUnavailable("x", retryable=False)),
        )
    assert _status(cli_env, job_a.job_id) == "failed"

    # B: superseded content -- must be skipped, retrying it would just fail again.
    for content in ("B v1", "B v2"):
        job = create_ingestion_job(
            cli_env,
            external_id="b.md",
            title="B",
            tier=Tier.GENERAL,
            content=content,
            embedding_model=FAKE_EMBEDDING_MODEL,
        )
        assert job.job_id is not None
        run_ingestion_job(cli_env, job.job_id, embed=fake_embed)
    with pytest.raises(SupersededContentError):
        create_ingestion_job(
            cli_env,
            external_id="b.md",
            title="B",
            tier=Tier.GENERAL,
            content="B v1",
            embedding_model=FAKE_EMBEDDING_MODEL,
        )
    with cli_env.connect() as conn:
        job_b = conn.execute(
            text("SELECT id, status FROM ingestion_jobs WHERE error_code = 'superseded_content'")
        ).one()

    assert main(["retry-failed"]) == 0

    assert _status(cli_env, job_a.job_id) == "completed"  # retried and completed
    assert _status(cli_env, job_b.id) == "failed"  # left untouched


def test_republish_stale_only_touches_undispatched_or_overdue_jobs(cli_env: Engine) -> None:
    with cli_env.begin() as conn:
        never_dispatched = _seed_queued_job(conn, "x.md", celery_task_id=None, overdue=False)
        overdue = _seed_queued_job(conn, "y.md", celery_task_id="t-1", overdue=True)
        fresh = _seed_queued_job(conn, "z.md", celery_task_id="t-2", overdue=False)

    assert main(["republish-stale"]) == 0

    assert _status(cli_env, never_dispatched) == "completed"
    assert _status(cli_env, overdue) == "completed"
    assert _status(cli_env, fresh) == "queued"  # dispatched recently; left alone


def _seed_queued_job(
    conn: Connection, external_id: str, *, celery_task_id: str | None, overdue: bool
) -> uuid.UUID:
    doc = make_document(conn, external_id)
    version = make_version(conn, doc, 1, "building", content=f"content for {external_id}")
    offset = "now() - interval '10 minutes'" if overdue else "now() + interval '1 hour'"
    job_id: uuid.UUID = conn.execute(
        text(
            "INSERT INTO ingestion_jobs (document_id, document_version_id, version_no, "
            "content_hash, idempotency_key, status, celery_task_id, available_at) "
            f"VALUES (:d, :v, 1, 'h', :k, 'queued', :t, {offset}) RETURNING id"
        ),
        {"d": doc, "v": version, "k": f"seed-{external_id}", "t": celery_task_id},
    ).scalar_one()
    return job_id


def test_cleanup_versions_keeps_last_n_across_all_statuses_and_drops_the_rest(
    cli_env: Engine,
) -> None:
    with cli_env.begin() as conn:
        doc = make_document(conn, "policies/many.md")
        for n in range(1, 5):
            v = make_version(conn, doc, n, "superseded", content=f"v{n}")
            conn.execute(
                text(
                    "UPDATE document_versions SET created_at = now() - interval '60 days' "
                    "WHERE id = :v"
                ),
                {"v": v},
            )
        v5 = make_version(conn, doc, 5, "active", content="v5")
        point_active(conn, doc, v5)
        make_chunk(conn, v5, 0, "chunk", [0.0] * 1536)

    assert main(["cleanup-versions", "--keep-last", "2", "--older-than-days", "30"]) == 0

    with cli_env.connect() as conn:
        remaining = dict(
            conn.execute(
                text("SELECT version_no, status FROM document_versions ORDER BY version_no")
            ).all()
        )
    # keep_last=2 counts across ALL versions: v5 (active, rank 1) and v4 (superseded, rank 2).
    assert remaining == {4: "superseded", 5: "active"}


def test_cleanup_versions_never_deletes_a_building_version(cli_env: Engine) -> None:
    with cli_env.begin() as conn:
        doc = make_document(conn, "policies/one.md")
        v1 = make_version(conn, doc, 1, "building", content="v1")
        conn.execute(
            text(
                "UPDATE document_versions SET created_at = now() - interval '90 days' "
                "WHERE id = :v"
            ),
            {"v": v1},
        )

    assert main(["cleanup-versions", "--keep-last", "0", "--older-than-days", "0"]) == 0

    with cli_env.connect() as conn:
        status = conn.execute(text("SELECT status FROM document_versions")).scalar_one()
    assert status == "building"
