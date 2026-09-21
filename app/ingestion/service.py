"""Ingestion service: the one place where ingestion logic lives (Plan §12.4, §12.6, §12.8).

WEEK 5 NOTE — read before adding the Celery task
------------------------------------------------
`ingest_document()` below is the REAL ingestion logic (version creation, chunk/embed, idempotency,
atomic activation). Week 5's `run_ingestion_job()` (the Celery task adapter in `tasks.py`) must
CALL `ingest_document()` / `activate_version()`; it must not re-implement, fork or wrap-and-diverge
from them under another name. The CLI calls the same functions (Plan §12.8: "CLI không được có
pipeline riêng khác Celery task"). What Week 5 adds around them is queue mechanics only: the
`ingestion_jobs` row, its status transitions, retry/backoff, and one extra
`UPDATE ingestion_jobs ... completed` statement inside the activation transaction (invariant 9).

WEEK 5 NOTE — open decision: SupersededContentError
---------------------------------------------------
When `ingest_document()` is wrapped by a Celery task and eventually exposed through the host
webhook, "content identical to a superseded version" (`SupersededContentError`) needs a decided
propagation to the host: HTTP 409? or `ingestion_jobs.status = 'failed'` plus a reason code? Not
decided in Week 2 on purpose — just do not forget it. Related, also left for Week 5: a stale build
(older version_no than the active one) is refused by `activate_version()` and its version is
marked `failed`; Plan §12.6 rule 7 wants the *job* marked `superseded`, which needs the job row.
"""
import hashlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Connection, Engine, text

from app.db.models import EMBEDDING_DIM
from app.ingestion.chunking import CHUNKING_VERSION, chunk_text
from app.ingestion.indexing import content_hash, index_config_hash, replace_chunks
from app.retrieval.schemas import Tier

_LOG = logging.getLogger(__name__)

# Injected so Week 2 needs no provider; Week 3's EmbeddingClient adapts to this shape.
# Must return vectors of exactly app.db.models.EMBEDDING_DIM floats, one per input text.
Embedder = Callable[[list[str]], list[list[float]]]


class IngestionError(Exception):
    """The document cannot be ingested; the active version (if any) is untouched."""


class SupersededContentError(IngestionError):
    """Content is identical to a superseded version. Re-activating it (rollback) is not a Week 2
    operation (Plan §12.7 / §12.6 rule 7), so it is refused rather than guessed at."""


class ActivationError(Exception):
    """Activation refused; nothing was changed."""


@dataclass(frozen=True)
class IngestResult:
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_no: int
    outcome: Literal["activated", "unchanged"]


def activate_version(engine: Engine, version_id: uuid.UUID) -> None:
    """Atomically make a fully-built `building` version the active one (invariant 9, Plan §5.2).

    Owns its transaction: supersede old -> activate new -> point documents.active_version_id ->
    knowledge_version += 1, all-or-nothing. Raises ActivationError (changing nothing) if the
    version is unknown, is not `building`, has no chunks, or is not newer than the active version.

    The old version must be demoted BEFORE the new one is promoted: the partial unique index
    `document_versions_one_active_per_document` is not deferrable and is checked per statement.
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
            raise ActivationError(
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
    same document are serialised with a session-level advisory lock, which the server releases if
    this process dies. On any failure after the version row exists, the version is marked `failed`
    and the previously active version keeps serving untouched.

    `embedding_model` names the model behind `embed`; it is part of index_config_hash, so changing
    it re-indexes unchanged content.
    """
    if not content.strip():
        raise IngestionError("document content is empty")
    chunks = chunk_text(content)
    key = _advisory_key(external_id)

    with engine.connect() as lock_conn:
        lock_conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": key})
        lock_conn.commit()  # session-level lock survives the commit; don't sit idle-in-transaction
        try:
            return _ingest_locked(
                engine,
                external_id=external_id,
                title=title,
                tier=tier,
                content=content,
                chunks=chunks,
                embed=embed,
                embedding_model=embedding_model,
            )
        finally:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                lock_conn.commit()
            except Exception:
                # A lock left on a pooled connection would wedge this document; drop the connection.
                _LOG.warning("advisory unlock failed; invalidating connection", exc_info=True)
                lock_conn.invalidate()


def _ingest_locked(
    engine: Engine,
    *,
    external_id: str,
    title: str,
    tier: Tier,
    content: str,
    chunks: list[str],
    embed: Embedder,
    embedding_model: str,
) -> IngestResult:
    c_hash = content_hash(content)
    cfg_hash = index_config_hash(embedding_model)

    with engine.begin() as conn:
        document_id = _get_or_create_document(conn, external_id, title, tier)
        existing = conn.execute(
            text(
                "SELECT id, version_no, status FROM document_versions "
                "WHERE document_id = :d AND content_hash = :h AND index_config_hash = :c"
            ),
            {"d": document_id, "h": c_hash, "c": cfg_hash},
        ).one_or_none()

        if existing is not None and existing.status == "active":
            return IngestResult(document_id, existing.id, existing.version_no, "unchanged")
        if existing is not None and existing.status == "superseded":
            raise SupersededContentError(
                f"{external_id}: content is identical to superseded version {existing.version_no}"
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

    try:
        vectors = embed(chunks)  # slow/remote in Week 3: deliberately outside any DB transaction
        if len(vectors) != len(chunks) or any(len(v) != EMBEDDING_DIM for v in vectors):
            raise IngestionError(
                f"embedder must return {len(chunks)} vectors of {EMBEDDING_DIM} floats"
            )
        with engine.begin() as conn:
            replace_chunks(conn, version_id, chunks, vectors)
        activate_version(engine, version_id)
    except Exception:
        _mark_failed(engine, version_id)
        raise
    return IngestResult(document_id, version_id, version_no, "activated")


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


def _mark_failed(engine: Engine, version_id: uuid.UUID) -> None:
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
