from celery import Celery

from app.settings import get_settings

# Week 1: broker wiring only. No tasks, no result backend (Plan §0: job state lives in
# PostgreSQL). Reliability settings (acks_late, reject_on_worker_lost, routing, retries)
# arrive with ingestion in Week 5 (Plan §12.5).
celery_app = Celery("rag_chatbot", broker=get_settings().rabbitmq_url.get_secret_value())
celery_app.conf.update(
    broker_connection_retry_on_startup=True,
    task_ignore_result=True,
)
