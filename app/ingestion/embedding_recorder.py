"""embedding_calls recording, offline (document/ingestion) path -- sync counterpart of
`app.ai.embedding_recorder`. A separate module, not a second implementation bolted onto the async
one: ingestion is sync throughout (`app.ingestion.service`), and unlike the online recorder, a
failed write here must NOT affect the ingestion outcome -- same "best effort" rule
`app.ingestion.service._mark_job`/`_mark_version_failed` already follow, because ingestion's real
correctness invariants (idempotency, atomic activation) must never be put at risk by an
observability write failing.
"""
import logging
import uuid

from sqlalchemy import Engine, text

from app.ai.embedding_recorder import EmbeddingCallRecord

_LOG = logging.getLogger(__name__)


class SqlSyncEmbeddingCallRecorder:
    def __init__(self, engine: Engine, ingestion_job_id: uuid.UUID) -> None:
        self._engine = engine
        self._job_id = ingestion_job_id

    def record(self, record: EmbeddingCallRecord) -> None:
        """Best effort: a failure recording the call must not mask -- or cause -- an ingestion
        failure."""
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO embedding_calls (ingestion_job_id, kind, provider, "
                        "model_version, batch_size, latency_ms, status, error_code) "
                        "VALUES (:j, :kind, :provider, :mv, :batch, :l, :status, :err)"
                    ),
                    {
                        "j": self._job_id,
                        "kind": record.kind,
                        "provider": record.provider,
                        "mv": record.model_version,
                        "batch": record.batch_size,
                        "l": record.latency_ms,
                        "status": record.status,
                        "err": record.error_code,
                    },
                )
        except Exception:
            _LOG.warning(
                "could not record embedding call for job %s", self._job_id, exc_info=True
            )
