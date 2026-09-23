"""Celery task adapter (Plan §12.4). Thin on purpose: all real logic lives in
`app.ingestion.service`, which this module must call, never reimplement (see that module's
docstring). A worker process builds its engine and embedder ONCE (lazily, on first task) and
reuses them for every task afterwards -- `SyncEmbedder` explicitly keeps one event loop for its
lifetime (see its docstring), so recreating it per task would break the second call.
"""
import logging
import uuid

from celery import Task
from sqlalchemy import Engine

from app.ai.embeddings.gemini import GeminiEmbeddingClient
from app.ai.embeddings.sync import SyncEmbedder
from app.db.session import create_sync_engine
from app.ingestion.service import TransientIngestionError, run_ingestion_job
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


@celery_app.task(  # type: ignore[untyped-decorator]  # celery is untyped (mypy override below)
    bind=True,
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
