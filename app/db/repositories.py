"""Small write helpers for turn bookkeeping (raw SQL, like app/retrieval/repository.py)."""
import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.retrieval.schemas import SourceSnapshot

if TYPE_CHECKING:
    from app.ai.recorder import ModelCallRecord


async def create_turn(engine: AsyncEngine, *, user_id: str, question: str) -> uuid.UUID:
    """One new conversation + one turn in status 'running'. The CLI is a one-turn conversation."""
    async with engine.begin() as conn:
        conversation_id = (
            await conn.execute(
                text("INSERT INTO conversations (user_id) VALUES (:u) RETURNING id"),
                {"u": user_id},
            )
        ).scalar_one()
        return (  # type: ignore[no-any-return]
            await conn.execute(
                text(
                    "INSERT INTO turns (conversation_id, question, graph_status) "
                    "VALUES (:c, :q, 'running') RETURNING id"
                ),
                {"c": conversation_id, "q": question},
            )
        ).scalar_one()


async def finish_turn(
    engine: AsyncEngine,
    turn_id: uuid.UUID,
    *,
    graph_status: str,
    answer: str | None,
    sources: Sequence[dict[str, str]],
    retrieval_attempts: int,
    latency_ms: int,
    snapshots: Sequence[SourceSnapshot] = (),
) -> None:
    """`answer` must only ever be a citation-validated answer (invariant #3).

    The turn update and its citation snapshots are ONE transaction: a turn is never left saying
    "answered" without the sources that let its citations be opened later (invariant #6).
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE turns SET graph_status = :s, answer = :a, "
                "sources_json = CAST(:src AS jsonb), retrieval_attempts = :r, latency_ms = :l "
                "WHERE id = :t"
            ),
            {"s": graph_status, "a": answer, "src": json.dumps(list(sources)),
             "r": retrieval_attempts, "l": latency_ms, "t": turn_id},
        )
        for snap in snapshots:
            await conn.execute(
                text(
                    "INSERT INTO turn_sources (turn_id, source_id, document_id, "
                    "document_version_id, chunk_id, document_title, version_no, text_snapshot, "
                    "metadata_snapshot) VALUES (:t, :sid, :d, :dv, :c, :title, :vn, :txt, "
                    "CAST(:meta AS jsonb))"
                ),
                {"t": turn_id, "sid": snap.source_id, "d": snap.document_id,
                 "dv": snap.document_version_id, "c": snap.chunk_id,
                 "title": snap.document_title, "vn": snap.version_no,
                 "txt": snap.text_snapshot, "meta": json.dumps(snap.metadata_snapshot)},
            )


async def get_turn_source(
    engine: AsyncEngine, turn_id: uuid.UUID, source_id: uuid.UUID
) -> SourceSnapshot | None:
    """Open a citation from its snapshot. Scoped by turn: another turn's source is not found.

    Reads only `turn_sources` — never the live document/chunk rows, which may be gone.
    """
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT source_id, document_id, document_version_id, chunk_id, "
                    "document_title, version_no, text_snapshot, metadata_snapshot "
                    "FROM turn_sources WHERE turn_id = :t AND source_id = :s"
                ),
                {"t": turn_id, "s": source_id},
            )
        ).mappings().first()
    if row is None:
        return None
    return SourceSnapshot(
        source_id=row["source_id"], document_id=row["document_id"],
        document_version_id=row["document_version_id"], chunk_id=row["chunk_id"],
        document_title=row["document_title"], version_no=row["version_no"],
        text_snapshot=row["text_snapshot"], metadata_snapshot=row["metadata_snapshot"],
    )


async def insert_model_call(
    engine: AsyncEngine, turn_id: uuid.UUID, record: "ModelCallRecord"
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO model_calls (turn_id, purpose, provider, model_name, prompt_version, "
                "input_tokens, output_tokens, latency_ms, status, error_code) "
                "VALUES (:t, :purpose, :provider, :model, :pv, :i, :o, :l, :status, :err)"
            ),
            {"t": turn_id, "purpose": record.purpose, "provider": record.provider,
             "model": record.model_name, "pv": record.prompt_version, "i": record.input_tokens,
             "o": record.output_tokens, "l": record.latency_ms, "status": record.status,
             "err": record.error_code},
        )
