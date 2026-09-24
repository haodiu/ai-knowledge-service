"""Week 8, e2e scenario #2 (Plan §16.3): provider timeout -> Celery retry -> job ends in the
correct state -- proven through the REAL `ingest_document_task` (`autoretry_for`, `retry_backoff`,
`max_retries=3`), not just the service-level `TransientIngestionError` classification
`test_ingestion_jobs.py` already covers.

Driven via `Task.apply()`, not `task_always_eager`: `.apply()` runs the task body synchronously,
in-process, with no broker connection, AND `Task.retry()`'s `is_eager` branch makes `.apply()`
recurse into itself on each retry (`retval.sig.apply(retries=retries + 1)`) -- so this exercises
Celery's real retry/backoff/max_retries bookkeeping, just without any actual queue delay (backoff's
`countdown` is only meaningful to `apply_async`'s broker scheduling, so `.apply()` never sleeps).
Matches CLAUDE.md's testing policy verbatim: "Celery runs in eager mode for logic tests; real
worker+RabbitMQ only for delivery-integration tests."
"""
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import Engine, text

from app.ai.errors import ModelTimeout, ModelUnavailable
from app.ingestion import tasks
from app.ingestion.fake_embedder import fake_embed
from app.ingestion.service import create_ingestion_job
from app.ingestion.tasks import ingest_document_task
from app.retrieval.schemas import Tier

pytestmark = pytest.mark.integration

CONTENT = "Refunds are possible within 30 days.\n\nContact support to start one."

# ingest_document_task.max_retries=3 -> 4 total attempts before Celery gives up (Plan §12.5).
MAX_RETRIES = ingest_document_task.max_retries
assert MAX_RETRIES == 3


def _create_job(engine: Engine) -> uuid.UUID:
    job = create_ingestion_job(
        engine,
        external_id="docs/refund.md",
        title="Refund policy",
        tier=Tier.GENERAL,
        content=CONTENT,
        embedding_model="fake-embedder",
    )
    assert job.job_id is not None
    return job.job_id


def _job_row(engine: Engine, job_id: uuid.UUID) -> dict[str, object]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status, error_code, retry_count FROM ingestion_jobs WHERE id = :j"),
            {"j": job_id},
        ).one()
    return dict(row._mapping)


@pytest.fixture
def patch_task_deps(
    monkeypatch: pytest.MonkeyPatch, db_engine: Engine
) -> Callable[[Callable[[list[str]], list[list[float]]]], None]:
    """Point the Celery task's lazily-cached engine/embedder at test doubles, so
    `ingest_document_task.apply()` runs its real body against the real test DB with no RabbitMQ or
    GEMINI_API_KEY needed. `_get_engine`/`_get_embedder` are plain module functions in
    `app.ingestion.tasks`, so monkeypatching them (not their cached globals) is enough."""

    def _patch(embed: Callable[[list[str]], list[list[float]]]) -> None:
        monkeypatch.setattr(tasks, "_get_engine", lambda: db_engine)
        monkeypatch.setattr(tasks, "_get_embedder", lambda: embed)

    return _patch


def test_transient_embedder_failure_retries_then_completes_via_celery(
    db_engine: Engine, patch_task_deps: Callable[[Any], None]
) -> None:
    job_id = _create_job(db_engine)
    calls = {"n": 0}

    def flaky(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ModelTimeout("simulated slow provider")
        return fake_embed(texts)

    patch_task_deps(flaky)

    result = ingest_document_task.apply(args=[str(job_id)])

    assert result.state == "SUCCESS"
    assert calls["n"] == 3  # 2 failures + 1 success, all via real Celery retries
    row = _job_row(db_engine, job_id)
    assert row["status"] == "completed"
    assert row["retry_count"] == 2


def test_transient_failure_exhausting_max_retries_ends_job_failed(
    db_engine: Engine, patch_task_deps: Callable[[Any], None]
) -> None:
    job_id = _create_job(db_engine)
    calls = {"n": 0}

    def always_down(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        raise ModelUnavailable("simulated 503", retryable=True)

    patch_task_deps(always_down)

    result = ingest_document_task.apply(args=[str(job_id)])

    assert result.state == "FAILURE"
    assert calls["n"] == MAX_RETRIES + 1  # 1 initial attempt + 3 retries, then Celery gives up
    row = _job_row(db_engine, job_id)
    assert row["status"] == "failed"
    assert row["error_code"] == "retries_exhausted"
    assert row["retry_count"] == MAX_RETRIES + 1


def test_permanent_failure_is_not_retried_by_celery(
    db_engine: Engine, patch_task_deps: Callable[[Any], None]
) -> None:
    job_id = _create_job(db_engine)
    calls = {"n": 0}

    def bad_request(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        raise ModelUnavailable("simulated bad request", retryable=False)

    patch_task_deps(bad_request)

    result = ingest_document_task.apply(args=[str(job_id)])

    assert result.state == "FAILURE"
    assert calls["n"] == 1  # no Celery retry spent on a non-transient failure
    row = _job_row(db_engine, job_id)
    assert row["status"] == "failed"
    assert row["error_code"] == "unavailable"  # the original code, not "retries_exhausted"
    assert row["retry_count"] == 0  # _process_locked only bumps retry_count on the transient path


def test_job_summary_is_logged_after_completion(
    db_engine: Engine,
    patch_task_deps: Callable[[Any], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B.3 (Plan §17): a structured summary line, carrying job_id/retry_count/queue_wait_ms/
    processing_time_ms, is emitted once processing actually finishes -- not just written to
    Postgres."""
    job_id = _create_job(db_engine)
    patch_task_deps(fake_embed)

    with caplog.at_level("INFO", logger="app.ingestion.tasks"):
        result = ingest_document_task.apply(args=[str(job_id)])
    assert result.state == "SUCCESS"

    summaries = [r for r in caplog.records if r.message == "ingestion job summary"]
    assert len(summaries) == 1
    assert summaries[0].job_id == str(job_id)
    assert summaries[0].celery_state == "SUCCESS"
    assert summaries[0].retry_count == 0
    assert summaries[0].processing_time_ms is not None
    assert summaries[0].processing_time_ms >= 0
    # Regression guard: queue_wait_ms must be measured from created_at, not available_at (a
    # republish-cooldown marker set 5 minutes into the future on dispatch, not the queue-entry
    # time) -- using the wrong column produced a large NEGATIVE value, caught by running this
    # against the real docker-compose stack.
    assert summaries[0].queue_wait_ms is not None
    assert summaries[0].queue_wait_ms >= 0


def test_document_row_reused_across_tests_does_not_leak(db_engine: Engine) -> None:
    """Guard against a fixture-ordering mistake: `db_engine` truncates `documents` per test
    (`tests/integration/conftest.py`), so two tests in this file that use the same `external_id`
    must not collide."""
    with db_engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM documents")).scalar_one()
    assert count == 0
