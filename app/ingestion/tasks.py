"""Celery task adapter (Plan §12.4). Thin on purpose: all real logic lives in
`app.ingestion.service`, which this module must call, never reimplement (see that module's
docstring). A worker process builds its engine and embedder ONCE (lazily, on first task) and
reuses them for every task afterwards -- `SyncEmbedder` explicitly keeps one event loop for its
lifetime (see its docstring), so recreating it per task would break the second call.
"""
import logging
import uuid
from typing import Any

from celery import Task
from celery.signals import task_postrun
from sqlalchemy import Engine, text

from app.ai.embeddings.gemini import GeminiEmbeddingClient
from app.ai.embeddings.sync import SyncEmbedder
from app.db.session import create_sync_engine
from app.ingestion.service import (
    TransientIngestionError,
    mark_job_retries_exhausted,
    run_ingestion_job,
)
from app.settings import get_settings
from app.worker.celery_app import celery_app

_LOG = logging.getLogger(__name__)

_engine: Engine | None = None
_embedder: SyncEmbedder | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_sync_engine(get_settings())
    return _engine


def _get_embedder() -> SyncEmbedder:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        if settings.gemini_api_key is None:
            raise RuntimeError("GEMINI_API_KEY is required to run the ingestion worker")
        _embedder = SyncEmbedder(
            GeminiEmbeddingClient(
                api_key=settings.gemini_api_key.get_secret_value(),
                timeout_seconds=settings.embedding_timeout_seconds,
            ),
            model_version=settings.embedding_model,
        )
    return _embedder


class _IngestTask(Task):  # type: ignore[misc]  # celery is untyped
    """Marks the job `failed` once Celery's own retries are exhausted (Plan §12.5) -- without this,
    a job whose transient failures never clear is stuck at `retrying` forever, since neither
    `run_ingestion_job()` nor Celery's retry machinery itself has anywhere else to make that
    terminal. `on_failure` receives the ORIGINAL exception Celery re-raises on exhaustion (verified
    against the installed celery==5.6.3: `Task.retry()`'s `raise_with_context(exc)` re-raises what
    `autoretry_for` caught, not a wrapped `MaxRetriesExceededError`), so the `isinstance` guard
    below is what distinguishes "retries exhausted" from a permanent failure -- the latter already
    self-marks the job `failed` inside `_process_locked()` and must not be double-handled here.
    """

    def on_failure(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        if isinstance(exc, TransientIngestionError):
            mark_job_retries_exhausted(_get_engine(), uuid.UUID(args[0]))


@celery_app.task(  # type: ignore[untyped-decorator]  # celery is untyped (mypy override below)
    bind=True,
    base=_IngestTask,
    name="app.ingestion.ingest_document",
    autoretry_for=(TransientIngestionError,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
    acks_late=True,
    reject_on_worker_lost=True,
)
def ingest_document_task(self: Task, job_id: str) -> None:
    """`self` (Celery's bound-task instance) is unused: `autoretry_for` drives the retry, so no
    manual `self.retry()` call is needed here."""
    run_ingestion_job(_get_engine(), uuid.UUID(job_id), embed=_get_embedder())


@task_postrun.connect(sender=ingest_document_task)  # type: ignore[untyped-decorator]
def _log_job_summary(
    task_id: str | None = None,
    args: tuple[Any, ...] = (),
    state: str | None = None,
    **kwargs: object,
) -> None:
    """One structured summary line per attempt (Plan §17: "Celery job_id, retry count, queue wait
    và processing time"). Fires on EVERY attempt, not just the terminal one -- `state` is `RETRY`
    for an attempt that will run again and `SUCCESS`/`FAILURE` for the last one (verified against
    celery==5.6.3: task_postrun fires with state='RETRY' mid-retry, then again at the end), so
    `queue_wait_ms`/`processing_time_ms` are naturally `None` until the job row actually has
    `started_at`/`completed_at` set. Best effort, same shape as `app.ingestion.service._mark_job`:
    a failure logging the summary must not affect the task's own outcome.
    """
    if not args:
        return
    try:
        job_id = uuid.UUID(args[0])
        with _get_engine().connect() as conn:
            row = conn.execute(
                text(
                    "SELECT created_at, started_at, completed_at, retry_count "
                    "FROM ingestion_jobs WHERE id = :j"
                ),
                {"j": job_id},
            ).one_or_none()
        if row is None:
            return
        # created_at, not available_at: available_at is a republish-COOLDOWN marker (app.
        # ingestion.cli._dispatch sets it to now() + 5 minutes on dispatch, so republish-stale
        # does not immediately re-select something just dispatched) -- it is not the time the job
        # became queued, and using it here produced a negative queue_wait_ms (caught by running
        # this against the real stack: a live ingest logged queue_wait_ms=-299728).
        queue_wait_ms = (
            int((row.started_at - row.created_at).total_seconds() * 1000)
            if row.started_at is not None
            else None
        )
        processing_time_ms = (
            int((row.completed_at - row.started_at).total_seconds() * 1000)
            if row.completed_at is not None and row.started_at is not None
            else None
        )
        _LOG.info(
            "ingestion job summary",
            extra={
                "job_id": str(job_id),
                "celery_task_id": task_id,
                "celery_state": state,
                "retry_count": row.retry_count,
                "queue_wait_ms": queue_wait_ms,
                "processing_time_ms": processing_time_ms,
            },
        )
    except Exception:
        _LOG.warning("could not log job summary for task %s", task_id, exc_info=True)
