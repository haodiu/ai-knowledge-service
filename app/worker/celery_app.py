from celery import Celery

from app.settings import get_settings

# No result backend (Plan §0: job state lives in PostgreSQL, in `ingestion_jobs` -- never in
# Celery's own result store). Reliability settings below are Plan §12.5, verbatim:
# - acks_late + reject_on_worker_lost: a task that crashes/loses its worker is redelivered, never
#   silently dropped. Safe because ingestion is idempotent (Plan §12.6, app.ingestion.service).
# - prefetch_multiplier=1: an ingestion task can run long (embedding a whole document); don't let
#   one worker hoard several off the queue while others sit idle.
INGESTION_QUEUE = "ingestion.default"

celery_app = Celery("rag_chatbot", broker=get_settings().rabbitmq_url.get_secret_value())
celery_app.conf.update(
    broker_connection_retry_on_startup=True,
    task_ignore_result=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_queue=INGESTION_QUEUE,
    worker_prefetch_multiplier=1,
)
# Lazy: Celery imports app.ingestion.tasks (which imports `celery_app` from this module) only once
# the app is finalized, not at module-load time -- avoids a circular import at startup.
celery_app.autodiscover_tasks(["app.ingestion"])
