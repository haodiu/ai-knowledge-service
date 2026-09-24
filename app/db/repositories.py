"""Small write helpers for turn bookkeeping (raw SQL, like app/retrieval/repository.py)."""
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.retrieval.schemas import SourceSnapshot

if TYPE_CHECKING:
    from app.ai.embedding_recorder import EmbeddingCallRecord
    from app.ai.recorder import ModelCallRecord
    from app.tools.recorder import ToolCallRecord


async def create_conversation(engine: AsyncEngine, user_id: str) -> uuid.UUID:
    async with engine.begin() as conn:
        return (  # type: ignore[no-any-return]
            await conn.execute(
                text("INSERT INTO conversations (user_id) VALUES (:u) RETURNING id"), {"u": user_id}
            )
        ).scalar_one()


async def conversation_owner(engine: AsyncEngine, conversation_id: uuid.UUID) -> str | None:
    """None if the conversation does not exist -- callers treat "not mine" and "not found" the
    same way (invariant #4: never disclose existence)."""
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT user_id FROM conversations WHERE id = :c"), {"c": conversation_id}
            )
        ).scalar_one_or_none()


async def add_turn(engine: AsyncEngine, conversation_id: uuid.UUID, question: str) -> uuid.UUID:
    """One new turn in status 'running', in an EXISTING conversation. Pair with
    `conversation_owner()` -- this does not check ownership itself."""
    async with engine.begin() as conn:
        return (  # type: ignore[no-any-return]
            await conn.execute(
                text(
                    "INSERT INTO turns (conversation_id, question, graph_status) "
                    "VALUES (:c, :q, 'running') RETURNING id"
                ),
                {"c": conversation_id, "q": question},
            )
        ).scalar_one()


@dataclass(frozen=True)
class CreatedTurn:
    conversation_id: uuid.UUID
    turn_id: uuid.UUID


async def create_turn(engine: AsyncEngine, *, user_id: str, question: str) -> CreatedTurn:
    """CLI convenience: one new conversation + one turn. Not required to be atomic across the two
    inserts -- a failure between them just leaves an unused, turn-less conversation row, the same
    as a host that calls POST /v1/conversations and never follows up with a turn."""
    conversation_id = await create_conversation(engine, user_id)
    turn_id = await add_turn(engine, conversation_id, question)
    return CreatedTurn(conversation_id, turn_id)


def _truncate(text_value: str, limit: int) -> str:
    return text_value if len(text_value) <= limit else text_value[: limit - 1] + "…"


async def get_recent_turns(
    engine: AsyncEngine, conversation_id: uuid.UUID, *, limit: int, max_chars_per_turn: int
) -> list[str]:
    """Up to `limit` most recent ANSWERED turns, oldest first (Plan §11.5). A turn with no answer
    yet -- including the turn currently `running` for this very request -- is excluded, so a turn
    can never see itself as its own history. `limit`/`max_chars_per_turn` are the caller's bounds
    (app.graph.limits.MAX_HISTORY_TURNS/MAX_HISTORY_CHARS_PER_TURN) -- kept as plain parameters
    here so this module stays independent of the graph package."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT question, answer FROM turns "
                    "WHERE conversation_id = :c AND answer IS NOT NULL "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"c": conversation_id, "limit": limit},
            )
        ).all()
    return [
        f"Q: {_truncate(r.question, max_chars_per_turn)}\n"
        f"A: {_truncate(r.answer, max_chars_per_turn)}"
        for r in reversed(rows)
    ]


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
    engine: AsyncEngine, turn_id: uuid.UUID, source_id: uuid.UUID, *, user_id: str
) -> SourceSnapshot | None:
    """Open a citation from its snapshot. Scoped by turn AND by ownership: another turn's source,
    or a turn belonging to a different user's conversation, is not found -- the same `None` either
    way (invariant #4: never disclose existence; a 403 would confirm the turn exists).

    Reads only `turn_sources` — never the live document/chunk rows, which may be gone.
    """
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT ts.source_id, ts.document_id, ts.document_version_id, ts.chunk_id, "
                    "ts.document_title, ts.version_no, ts.text_snapshot, ts.metadata_snapshot "
                    "FROM turn_sources ts "
                    "JOIN turns t ON t.id = ts.turn_id "
                    "JOIN conversations c ON c.id = t.conversation_id "
                    "WHERE ts.turn_id = :t AND ts.source_id = :s AND c.user_id = :u"
                ),
                {"t": turn_id, "s": source_id, "u": user_id},
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


async def insert_tool_call(
    engine: AsyncEngine, turn_id: uuid.UUID, record: "ToolCallRecord"
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO tool_calls (turn_id, tool_name, identifier_kind, latency_ms, "
                "status, error_code) VALUES (:t, :name, :kind, :l, :status, :err)"
            ),
            {"t": turn_id, "name": record.tool_name, "kind": record.identifier_kind,
             "l": record.latency_ms, "status": record.status, "err": record.error_code},
        )


async def insert_embedding_call(
    engine: AsyncEngine, *, turn_id: uuid.UUID, record: "EmbeddingCallRecord"
) -> None:
    """Online (query-embedding) path only -- the offline/ingestion path uses the sync
    `app.ingestion.embedding_recorder.SqlSyncEmbeddingCallRecorder` instead, since
    `app.ingestion.service` is sync throughout."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO embedding_calls (turn_id, kind, provider, model_version, "
                "batch_size, latency_ms, status, error_code) "
                "VALUES (:t, :kind, :provider, :mv, :batch, :l, :status, :err)"
            ),
            {
                "t": turn_id, "kind": record.kind, "provider": record.provider,
                "mv": record.model_version, "batch": record.batch_size,
                "l": record.latency_ms, "status": record.status, "err": record.error_code,
            },
        )


async def get_turn_usage(engine: AsyncEngine, turn_id: uuid.UUID) -> tuple[int, int, int]:
    """(call_count, total_input_tokens, total_output_tokens) for a turn's model_calls rows
    (Plan §17 token/cost observability). Summed from what insert_model_call() already wrote --
    no separate cost table."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT count(*), coalesce(sum(input_tokens), 0), "
                    "coalesce(sum(output_tokens), 0) FROM model_calls WHERE turn_id = :t"
                ),
                {"t": turn_id},
            )
        ).one()
    return row[0], row[1], row[2]
