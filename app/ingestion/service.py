"""Ingestion service: the one place where ingestion logic lives (Plan §12.4, §12.6, §12.8, §5.3).

Two lock-acquiring entrypoints share the same two locked building blocks
(`_create_job_locked()`, `_process_locked()`), so there is exactly one ingestion pipeline:

- `ingest_document()`: acquires one advisory lock and holds it across BOTH job creation and
  processing. This is the synchronous do-everything entrypoint (CLI default mode, tests, live
  tests). Holding one lock across both steps reproduces the pre-Week-5 behaviour under concurrent
  duplicate delivery: exactly one caller does the work and returns "activated"; the rest see the
  now-active version inside the lock and return "unchanged" without processing anything.
- `create_ingestion_job()` + `run_ingestion_job()`: each acquires and releases its OWN lock. Used
  by the CLI's `--queue` mode and the Celery task (`app/ingestion/tasks.py`), where creation and
  processing genuinely happen in two different calls -- possibly in two different processes -- so
  they cannot share a single in-process lock scope. Processing is idempotent (Plan §12.6 rule 1):
  a job already in a terminal state is a no-op, and `activate_version()` completes the job
  atomically with activation (invariant 9), so redelivery after a crash can never double-apply.

`run_ingestion_job()` is the function the Celery task adapter calls; `tasks.py` must call it, not
reimplement it (Plan §12.8: "CLI không được có pipeline riêng khác Celery task" applies here too).
"""
import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import OperationalError

from app.ai.embedding_recorder import EmbeddingCallRecord
from app.ai.errors import ModelTimeout, ModelUnavailable
from app.db.models import EMBEDDING_DIM
from app.ingestion.chunking import CHUNKING_VERSION, chunk_text
from app.ingestion.embedding_recorder import SqlSyncEmbeddingCallRecorder
from app.ingestion.indexing import content_hash, index_config_hash, replace_chunks
from app.retrieval.schemas import Tier

_LOG = logging.getLogger(__name__)

# Injected so Week 2 needs no provider; Week 3's EmbeddingClient adapts to this shape.
# Must return vectors of exactly app.db.models.EMBEDDING_DIM floats, one per input text.
Embedder = Callable[[list[str]], list[list[float]]]

_TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "superseded"})


class IngestionError(Exception):
    """The document cannot be ingested; the active version (if any) is untouched."""


class SupersededContentError(IngestionError):
    """Content is identical to a superseded version. Re-activating it (rollback) is not a Week 2
    operation (Plan §12.7 / §12.6 rule 7), so it is refused rather than guessed at.

    `job_id`/`version_no` (Week 6) describe the `failed`/`superseded_content` audit job row that
    was already committed before this was raised (see `_create_job_locked`) -- the HTTP ingestion
    endpoint uses them to still return 202 + a job the caller can poll, per the Week 5 decision
    not to give this its own HTTP error branch."""

    def __init__(
        self, message: str = "", *, job_id: uuid.UUID | None = None, version_no: int = 0
    ) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.version_no = version_no


class ActivationError(Exception):
    """Activation refused; nothing was changed."""


class StaleVersionError(ActivationError):
    """The building version is not newer than the currently active one -- two jobs raced for the
    same document and this one lost (Plan §12.6 rule 7). A distinct subtype (not just a message)
    so `run_ingestion_job()` can mark the *job* `superseded` rather than `failed` without string-
    matching. `pytest.raises(ActivationError)` still catches it; nothing that only cared about the
    base class needs to change."""


class JobNotFoundError(Exception):
    """`run_ingestion_job()`/`create_ingestion_job()` internals were given an unknown job_id."""


class TransientIngestionError(Exception):
    """A retryable failure occurred while processing an already-queued job (Plan §12.5): the job
    row is marked `retrying` before this is raised, wrapping the original error as `__cause__`.
    Celery's `ingest_document_task` (app/ingestion/tasks.py) retries on this specific type via
    `autoretry_for`. Called outside Celery (CLI sync mode, tests), it is just another exception --
    there is no special handling for it here."""


@dataclass(frozen=True)
class IngestResult:
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_no: int
    outcome: Literal["activated", "unchanged"]
    job_id: uuid.UUID | None = None  # None for "unchanged": no job is created for a no-op ingest


@dataclass(frozen=True)
class JobRef:
    """What `_create_job_locked()` hands back internally. `job_id` is None only for "completed"
    (content matched an already-active version: nothing to track -- no job row exists). For
    "rejected" (content matched a superseded version), `job_id` points at the `failed`/
    `superseded_content` audit row that was already written inside the same transaction.
    "rejected" only ever appears inside this module: `create_ingestion_job()`/`ingest_document()`
    turn it into a raised `SupersededContentError(job_id=...)` immediately after their transaction
    commits -- raising while still inside `engine.begin()` would roll back the audit insert."""

    job_id: uuid.UUID | None
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_no: int
    status: Literal["queued", "completed", "rejected"]
    rejection_reason: str | None = None


def activate_version(
    engine: Engine, version_id: uuid.UUID, *, job_id: uuid.UUID | None = None
) -> None:
    """Atomically make a fully-built `building` version the active one (invariant 9, Plan §5.2).

    Owns its transaction: supersede old -> activate new -> point documents.active_version_id ->
    knowledge_version += 1 -> (if `job_id` given) complete the job, all-or-nothing. Completing the
    job in the SAME transaction as activation is invariant 9's "complete the job" step: it closes
    the window where a crash right after activation would otherwise leave a job stuck `processing`
    even though its version is already serving traffic.

    Raises ActivationError (changing nothing) if the version is unknown, is not `building`, or has
    no chunks; raises StaleVersionError (an ActivationError subclass) if it is not newer than the
    active version.
    """
    with engine.begin() as conn:
        document_id = conn.execute(
            text("SELECT document_id FROM document_versions WHERE id = :v"), {"v": version_id}
        ).scalar_one_or_none()
        if document_id is None:
            raise ActivationError(f"unknown version {version_id}")

        # Lock order: document row, then its version row. Serialises concurrent activations.
        conn.execute(text("SELECT id FROM documents WHERE id = :d FOR UPDATE"), {"d": document_id})
        version = conn.execute(
            text("SELECT status, version_no FROM document_versions WHERE id = :v FOR UPDATE"),
            {"v": version_id},
        ).one()
        if version.status != "building":
            raise ActivationError(f"version {version_id} is '{version.status}', not 'building'")
        has_chunks = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM chunks WHERE document_version_id = :v)"),
            {"v": version_id},
        ).scalar_one()
        if not has_chunks:
            raise ActivationError(f"version {version_id} has no chunks")
        active_no = conn.execute(
            text(
                "SELECT version_no FROM document_versions "
                "WHERE document_id = :d AND status = 'active'"
            ),
            {"d": document_id},
        ).scalar_one_or_none()
        if active_no is not None and active_no >= version.version_no:
            raise StaleVersionError(
                f"version {version.version_no} is not newer than active version {active_no}"
            )

        conn.execute(
            text(
                "UPDATE document_versions SET status = 'superseded' "
                "WHERE document_id = :d AND status = 'active'"
            ),
            {"d": document_id},
        )
        _expect_one_row(
            conn.execute(
                text(
                    "UPDATE document_versions SET status = 'active', activated_at = now() "
                    "WHERE id = :v AND status = 'building'"
                ),
                {"v": version_id},
            ).rowcount,
            "promote version",
        )
        _expect_one_row(
            conn.execute(
                text(
                    "UPDATE documents SET active_version_id = :v, updated_at = now() WHERE id = :d"
                ),
                {"v": version_id, "d": document_id},
            ).rowcount,
            "point document at version",
        )
        _expect_one_row(
            conn.execute(
                text(
                    "UPDATE knowledge_base_state "
                    "SET knowledge_version = knowledge_version + 1, updated_at = now() "
                    "WHERE singleton_id = 1"
                )
            ).rowcount,
            "bump knowledge_version",
        )
        if job_id is not None:
            _expect_one_row(
                conn.execute(
                    text(
                        "UPDATE ingestion_jobs SET status = 'completed', completed_at = now(), "
                        "document_version_id = :v "
                        "WHERE id = :j AND status IN ('processing', 'retrying')"
                    ),
                    {"v": version_id, "j": job_id},
                ).rowcount,
                "complete ingestion job",
            )


def _expect_one_row(rowcount: int, what: str) -> None:
    if rowcount != 1:  # raising inside engine.begin() rolls the whole activation back
        raise ActivationError(f"{what}: expected to update 1 row, updated {rowcount}")


def _advisory_key(external_id: str) -> int:
    """Stable signed 64-bit key for pg_advisory_lock, one per document."""
    digest = hashlib.sha256(external_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def ingest_document(
    engine: Engine,
    *,
    external_id: str,
    title: str,
    tier: Tier,
    content: str,
    embed: Embedder,
    embedding_model: str,
) -> IngestResult:
    """Chunk + embed `content` into a new version and activate it; no-op if already active.

    Idempotent (Plan §12.6): keyed by (document, content_hash, index_config_hash). Ingests of the
    same document are serialised with a session-level advisory lock, held across BOTH job creation
    and processing (see module docstring), which the server releases if this process dies. On any
    failure after the version row exists, the version is marked `failed` and the previously active
    version keeps serving untouched.

    `embedding_model` names the model behind `embed`; it is part of index_config_hash, so changing
    it re-indexes unchanged content.
    """
    if not content.strip():
        raise IngestionError("document content is empty")
    key = _advisory_key(external_id)

    with engine.connect() as lock_conn:
        lock_conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": key})
        lock_conn.commit()  # session-level lock survives the commit; don't sit idle-in-transaction
        try:
            with engine.begin() as conn:
                job = _create_job_locked(
                    conn,
                    external_id=external_id,
                    title=title,
                    tier=tier,
                    content=content,
                    embedding_model=embedding_model,
                )
            if job.status == "rejected":
                assert job.rejection_reason is not None
                raise SupersededContentError(
                    job.rejection_reason, job_id=job.job_id, version_no=job.version_no
                )
            if job.status == "completed":
                return IngestResult(
                    job.document_id, job.version_id, job.version_no, "unchanged", job.job_id
                )
            assert job.job_id is not None  # status == "queued" always has a job_id
            _process_locked(engine, job.job_id, embed=embed)
            final_status = _job_status(engine, job.job_id)
            if final_status != "completed":
                # Only reachable if a concurrent process raced this exact job to a different
                # terminal state under the same lock hand-off; genuine processing failures raise
                # out of _process_locked() above instead of returning here.
                raise IngestionError(f"{external_id}: job ended as {final_status!r}, not completed")
            return IngestResult(
                job.document_id, job.version_id, job.version_no, "activated", job.job_id
            )
        finally:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                lock_conn.commit()
            except Exception:
                # A lock left on a pooled connection would wedge this document; drop the connection.
                _LOG.warning("advisory unlock failed; invalidating connection", exc_info=True)
                lock_conn.invalidate()


def create_ingestion_job(
    engine: Engine,
    *,
    external_id: str,
    title: str,
    tier: Tier,
    content: str,
    embedding_model: str,
) -> JobRef:
    """Create (or idempotently reuse) an `ingestion_jobs` row for `content`, without processing it.

    For callers that hand processing off to a worker instead of doing it inline: the CLI's
    `--queue` mode and, once it exists, the HTTP ingestion endpoint. Pair with
    `run_ingestion_job()` to actually do the work. `ingest_document()` covers the synchronous
    create-and-process case and should be preferred when there is no reason to split the two.

    Raises SupersededContentError, exactly as `ingest_document()` does, if `content` is identical
    to a version that has already been superseded.
    """
    if not content.strip():
        raise IngestionError("document content is empty")
    key = _advisory_key(external_id)

    with engine.connect() as lock_conn:
        lock_conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": key})
        lock_conn.commit()
        try:
            with engine.begin() as conn:
                job = _create_job_locked(
                    conn,
                    external_id=external_id,
                    title=title,
                    tier=tier,
                    content=content,
                    embedding_model=embedding_model,
                )
            if job.status == "rejected":
                assert job.rejection_reason is not None
                raise SupersededContentError(
                    job.rejection_reason, job_id=job.job_id, version_no=job.version_no
                )
            return job
        finally:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                lock_conn.commit()
            except Exception:
                _LOG.warning("advisory unlock failed; invalidating connection", exc_info=True)
                lock_conn.invalidate()


def run_ingestion_job(engine: Engine, job_id: uuid.UUID, *, embed: Embedder) -> None:
    """Process a queued/retrying job to completion: chunk, embed, write chunks, activate.

    Idempotent (Plan §12.6 rule 1): a job already in a terminal state (`completed`/`failed`/
    `superseded`) is a no-op. This is what the Celery task adapter (`app/ingestion/tasks.py`)
    calls, and what the CLI's `--queue` mode drives synchronously without Celery in tests.
    """
    status = _job_status(engine, job_id)
    if status is None:
        raise JobNotFoundError(str(job_id))
    if status in _TERMINAL_JOB_STATUSES:
        return
    ctx = _load_job_context(engine, job_id)
    key = _advisory_key(ctx.external_id)

    with engine.connect() as lock_conn:
        lock_conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": key})
        lock_conn.commit()
        try:
            _process_locked(engine, job_id, embed=embed)
        finally:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                lock_conn.commit()
            except Exception:
                _LOG.warning("advisory unlock failed; invalidating connection", exc_info=True)
                lock_conn.invalidate()


def _create_job_locked(
    conn: Connection,
    *,
    external_id: str,
    title: str,
    tier: Tier,
    content: str,
    embedding_model: str,
) -> JobRef:
    c_hash = content_hash(content)
    cfg_hash = index_config_hash(embedding_model)
    document_id = _get_or_create_document(conn, external_id, title, tier)
    existing = conn.execute(
        text(
            "SELECT id, version_no, status FROM document_versions "
            "WHERE document_id = :d AND content_hash = :h AND index_config_hash = :c"
        ),
        {"d": document_id, "h": c_hash, "c": cfg_hash},
    ).one_or_none()

    if existing is not None and existing.status == "active":
        # Nothing to do or track: this is a no-op resubmission of what is already live.
        return JobRef(None, document_id, existing.id, existing.version_no, "completed")

    if existing is not None and existing.status == "superseded":
        # A terminal, one-off record of the rejected attempt -- not something later redelivery
        # needs to be deduplicated against, so the key just needs to be unique, not deterministic.
        # document_version_id is left NULL: it is UNIQUE, and the superseded version's slot is
        # already taken by the job that originally built it -- this row is an audit trail of a
        # *rejected* attempt, not a second job for that same version. version_no (a plain int, no
        # FK) still records which version the rejected content matched.
        reject_key = (
            f"{document_id}:{existing.version_no}:{c_hash}:{cfg_hash}:rejected:{uuid.uuid4().hex}"
        )
        reject_job_id = conn.execute(
            text(
                "INSERT INTO ingestion_jobs (document_id, document_version_id, version_no, "
                "content_hash, idempotency_key, status, error_code, completed_at) "
                "VALUES (:d, NULL, :n, :h, :k, 'failed', 'superseded_content', now()) "
                "RETURNING id"
            ),
            {
                "d": document_id,
                "n": existing.version_no,
                "h": c_hash,
                "k": reject_key,
            },
        ).scalar_one()
        # Do NOT raise here: this is still inside the caller's `engine.begin()`, and raising out
        # of that block would roll back the audit INSERT just above. Report "rejected" instead;
        # the caller raises SupersededContentError once its transaction has committed.
        return JobRef(
            reject_job_id,
            document_id,
            existing.id,
            existing.version_no,
            "rejected",
            rejection_reason=(
                f"{external_id}: content is identical to superseded version {existing.version_no}"
            ),
        )

    if existing is not None:
        # building/failed: rebuild the same row (the unique key forbids inserting a new one)
        version_id, version_no = existing.id, existing.version_no
        conn.execute(
            text("UPDATE document_versions SET status = 'building' WHERE id = :v"),
            {"v": version_id},
        )
    else:
        version_no = conn.execute(
            text(
                "SELECT coalesce(max(version_no), 0) + 1 FROM document_versions "
                "WHERE document_id = :d"
            ),
            {"d": document_id},
        ).scalar_one()
        version_id = conn.execute(
            text(
                "INSERT INTO document_versions "
                "(document_id, version_no, content, content_hash, chunking_version, "
                "embedding_model, index_config_hash, status) "
                "VALUES (:d, :n, :content, :h, :cv, :em, :c, 'building') RETURNING id"
            ),
            {
                "d": document_id,
                "n": version_no,
                "content": content,
                "h": c_hash,
                "cv": CHUNKING_VERSION,
                "em": embedding_model,
                "c": cfg_hash,
            },
        ).scalar_one()

    idem_key = f"{document_id}:{version_no}:{c_hash}:{cfg_hash}"
    job_id = _find_or_recycle_job(
        conn,
        document_id=document_id,
        version_id=version_id,
        version_no=version_no,
        content_hash_value=c_hash,
        idempotency_key=idem_key,
    )
    return JobRef(job_id, document_id, version_id, version_no, "queued")


def _find_or_recycle_job(
    conn: Connection,
    *,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    version_no: int,
    content_hash_value: str,
    idempotency_key: str,
) -> uuid.UUID:
    """Find the job already tracking this exact (document, version, content, config) attempt, or
    create one. An in-flight job (`queued`/`processing`/`retrying`) is returned untouched -- this
    is the redelivery-dedup case duplicate Celery/CLI submissions must hit (Plan §12.6 rule 1: at
    most one job actually does the work). Anything else found (a stale `failed`/`superseded`
    row -- `completed` cannot legitimately coexist with a `building`/`failed` version, but is
    handled the same way defensively) is recycled to `queued`, mirroring how a retry rebuilds the
    same version row rather than creating a new one.
    """
    existing = conn.execute(
        text("SELECT id, status FROM ingestion_jobs WHERE idempotency_key = :k"),
        {"k": idempotency_key},
    ).one_or_none()
    if existing is None:
        job_id: uuid.UUID = conn.execute(
            text(
                "INSERT INTO ingestion_jobs (document_id, document_version_id, version_no, "
                "content_hash, idempotency_key, status) VALUES (:d, :v, :n, :h, :k, 'queued') "
                "RETURNING id"
            ),
            {
                "d": document_id,
                "v": version_id,
                "n": version_no,
                "h": content_hash_value,
                "k": idempotency_key,
            },
        ).scalar_one()
        return job_id
    if existing.status in ("queued", "processing", "retrying"):
        in_flight_id: uuid.UUID = existing.id  # already in flight; another caller owns it
        return in_flight_id
    conn.execute(
        text(
            "UPDATE ingestion_jobs SET status = 'queued', error_code = NULL, "
            "error_message = NULL, document_version_id = :v, completed_at = NULL, "
            "retry_count = retry_count + 1 WHERE id = :j"
        ),
        {"v": version_id, "j": existing.id},
    )
    job_id = existing.id
    return job_id


@dataclass(frozen=True)
class _JobContext:
    version_id: uuid.UUID
    external_id: str
    content: str
    embedding_model: str


def _job_status(engine: Engine, job_id: uuid.UUID) -> str | None:
    with engine.connect() as conn:
        status: str | None = conn.execute(
            text("SELECT status FROM ingestion_jobs WHERE id = :j"), {"j": job_id}
        ).scalar_one_or_none()
    return status


def _load_job_context(engine: Engine, job_id: uuid.UUID) -> _JobContext:
    """Only called once the job is known non-terminal, so `document_version_id` is guaranteed
    non-null: cleanup (Plan §12.7) never touches a `building` version, which is what a queued/
    processing/retrying job always points at."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT dv.id AS version_id, d.external_id, dv.content, dv.embedding_model "
                "FROM ingestion_jobs j "
                "JOIN documents d ON d.id = j.document_id "
                "JOIN document_versions dv ON dv.id = j.document_version_id "
                "WHERE j.id = :j"
            ),
            {"j": job_id},
        ).one()
    return _JobContext(row.version_id, row.external_id, row.content, row.embedding_model)


def _process_locked(engine: Engine, job_id: uuid.UUID, *, embed: Embedder) -> None:
    # Re-check after acquiring the lock: another process may have just finished this job while
    # this caller was waiting for it.
    status = _job_status(engine, job_id)
    if status is None or status in _TERMINAL_JOB_STATUSES:
        return
    ctx = _load_job_context(engine, job_id)
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE ingestion_jobs SET status = 'processing', "
                "started_at = coalesce(started_at, now()) WHERE id = :j"
            ),
            {"j": job_id},
        )
        # A prior attempt on THIS SAME job may have left the version 'failed' (see the transient
        # branch below): activate_version() only accepts a 'building' one, so a retry -- automatic
        # (Celery's autoretry_for) or manual (app.ingestion.cli's retry-failed, which does the
        # equivalent reset explicitly) -- must reset it back before trying again. A no-op when the
        # version is already 'building' (the common case: first attempt, or superseded-by-a-newer-
        # job never reaches here since that job is already terminal).
        conn.execute(
            text(
                "UPDATE document_versions SET status = 'building' "
                "WHERE id = :v AND status = 'failed'"
            ),
            {"v": ctx.version_id},
        )

    chunks = chunk_text(ctx.content)
    embed_recorder = SqlSyncEmbeddingCallRecorder(engine, job_id)
    # Not every Embedder (the plain Callable[[list[str]], list[list[float]]] this module expects)
    # self-identifies -- only ones wrapping a real EmbeddingClient (app.ai.embeddings.sync.
    # SyncEmbedder) do. A bare test function (e.g. app.ingestion.fake_embedder.fake_embed) records
    # as "unknown" rather than a guessed provider name.
    embed_provider = getattr(embed, "provider", "unknown")
    try:
        started = time.perf_counter()
        try:
            vectors = embed(chunks)  # slow/remote: deliberately outside any DB transaction
        except Exception as exc:
            embed_recorder.record(
                EmbeddingCallRecord(
                    "document", embed_provider, ctx.embedding_model, len(chunks),
                    int((time.perf_counter() - started) * 1000), "error", _error_code(exc),
                )
            )
            raise
        embed_recorder.record(
            EmbeddingCallRecord(
                "document", embed_provider, ctx.embedding_model, len(chunks),
                int((time.perf_counter() - started) * 1000), "ok", None,
            )
        )
        if len(vectors) != len(chunks) or any(len(v) != EMBEDDING_DIM for v in vectors):
            raise IngestionError(
                f"embedder must return {len(chunks)} vectors of {EMBEDDING_DIM} floats"
            )
        with engine.begin() as conn:
            replace_chunks(conn, ctx.version_id, chunks, vectors)
        activate_version(engine, ctx.version_id, job_id=job_id)
    except StaleVersionError as exc:
        _mark_version_failed(engine, ctx.version_id)
        _mark_job(engine, job_id, status="superseded")
        _LOG.info("job %s superseded: %s", job_id, exc)
    except Exception as exc:
        _mark_version_failed(engine, ctx.version_id)
        if _is_transient(exc):
            _mark_job(
                engine, job_id, status="retrying", error_code=_error_code(exc), bump_retry=True
            )
            raise TransientIngestionError(str(exc)) from exc
        _mark_job(engine, job_id, status="failed", error_code=_error_code(exc))
        raise


def _is_transient(exc: Exception) -> bool:
    """Fail closed: only recognised, genuinely-transient causes get a Celery retry (Plan §12.5:
    "lỗi validation/OCR không hỗ trợ không được retry vô ích"). An unrecognised exception type is
    treated as permanent, so a real bug surfaces once instead of retry-looping three times."""
    if isinstance(exc, ModelTimeout):
        return True
    if isinstance(exc, ModelUnavailable):
        return exc.retryable
    if isinstance(exc, OperationalError):  # DB connection lost/refused/timed out
        return True
    return False


def _error_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) else type(exc).__name__


def _mark_job(
    engine: Engine,
    job_id: uuid.UUID,
    *,
    status: str,
    error_code: str | None = None,
    bump_retry: bool = False,
) -> None:
    """Best effort: a failure recording the failure must not mask the original error."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE ingestion_jobs SET status = :s, error_code = :ec, "
                    "retry_count = retry_count + :bump, completed_at = "
                    "CASE WHEN :s IN ('failed', 'superseded') THEN now() ELSE completed_at END "
                    "WHERE id = :j"
                ),
                {"s": status, "ec": error_code, "bump": 1 if bump_retry else 0, "j": job_id},
            )
    except Exception:
        _LOG.warning("could not mark job %s as %s", job_id, status, exc_info=True)


def mark_job_retries_exhausted(
    engine: Engine, job_id: uuid.UUID, *, error_code: str = "retries_exhausted"
) -> None:
    """Terminal transition for a job whose Celery retries are exhausted (Plan §12.5).

    `_process_locked()` marks each transient attempt `retrying` (bumping `retry_count`) and raises
    `TransientIngestionError`; nothing transitions the job to `failed` once
    `app.ingestion.tasks.ingest_document_task`'s `max_retries` is used up, so without this a job
    whose transient failures never clear is stuck at `retrying` forever. Called from that task's
    `on_failure` hook, only when the final exception is a `TransientIngestionError` -- a permanent
    failure already self-marks `failed` inside `_process_locked()` and must not be double-handled.
    Best effort, same shape as `_mark_job`: a failure recording the failure must not mask the
    original error.
    """
    _mark_job(engine, job_id, status="failed", error_code=error_code)


def _get_or_create_document(
    conn: Connection, external_id: str, title: str, tier: Tier
) -> uuid.UUID:
    created: uuid.UUID | None = conn.execute(
        text(
            "INSERT INTO documents (external_id, title, tier) VALUES (:e, :t, :tier) "
            "ON CONFLICT (external_id) DO NOTHING RETURNING id"
        ),
        {"e": external_id, "t": title, "tier": tier.value},
    ).scalar_one_or_none()
    if created is not None:
        return created

    row = conn.execute(
        text("SELECT id, tier, status FROM documents WHERE external_id = :e"), {"e": external_id}
    ).one()
    if row.status != "active":
        raise IngestionError(f"{external_id}: document is '{row.status}', not accepting versions")
    if row.tier != tier.value:
        # tier lives on `documents`, so applying it now would re-scope the CURRENTLY ACTIVE
        # version immediately (e.g. internal -> general leaks old content). Refuse; fail closed.
        raise IngestionError(
            f"{external_id}: tier change {row.tier!r} -> {tier.value!r} is not supported"
        )
    conn.execute(
        text("UPDATE documents SET title = :t, updated_at = now() WHERE id = :d AND title <> :t"),
        {"t": title, "d": row.id},
    )
    document_id: uuid.UUID = row.id
    return document_id


def _mark_version_failed(engine: Engine, version_id: uuid.UUID) -> None:
    """Best effort: a failure here must not mask the original error."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE document_versions SET status = 'failed' "
                    "WHERE id = :v AND status = 'building'"
                ),
                {"v": version_id},
            )
    except Exception:
        _LOG.warning("could not mark version %s failed", version_id, exc_info=True)
