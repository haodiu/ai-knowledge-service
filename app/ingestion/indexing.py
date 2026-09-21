"""Hashing and chunk persistence for ingestion (Plan §12.6)."""
import hashlib
import json
import uuid

from sqlalchemy import Connection, text

from app.db.models import EMBEDDING_DIM
from app.db.vectors import to_vector_literal
from app.ingestion.chunking import CHUNKING_VERSION, MAX_CHUNK_CHARS


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def index_config_hash(embedding_model: str) -> str:
    """Everything except the content that changes what the index looks like (Plan §5.4).

    Covers the chunking configuration and the embedding model + dimensions, so identical content is
    re-indexed when any of them changes.
    """
    config = {
        "chunking_version": CHUNKING_VERSION,
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "embedding_model": embedding_model,
        "embedding_dim": EMBEDDING_DIM,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def replace_chunks(
    conn: Connection,
    version_id: uuid.UUID,
    chunks: list[str],
    vectors: list[list[float]],
) -> None:
    """Write all chunks of a version, replacing any partial set from an earlier attempt.

    Runs in the caller's transaction, so either every chunk lands or none does.
    UNIQUE(document_version_id, chunk_index) is the DB-level backstop against duplicate delivery.
    """
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    conn.execute(text("DELETE FROM chunks WHERE document_version_id = :v"), {"v": version_id})
    conn.execute(
        text(
            "INSERT INTO chunks (document_version_id, chunk_index, text, embedding) "
            "VALUES (:v, :i, :t, CAST(:e AS vector))"
        ),
        [
            {"v": version_id, "i": i, "t": body, "e": to_vector_literal(vec)}
            for i, (body, vec) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )
