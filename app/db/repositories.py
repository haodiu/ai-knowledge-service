"""Small write helpers for turn bookkeeping (raw SQL, like app/retrieval/repository.py)."""
import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

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
) -> None:
    """`answer` must only ever be a citation-validated answer (invariant #3)."""
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
