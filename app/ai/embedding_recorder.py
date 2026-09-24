"""embedding_calls recording, online (query) path -- operational metadata only, mirrors
`app/ai/recorder.py`. Kept as a sibling module, not folded into it: embeddings are a different
protocol/lifecycle from chat calls (CLAUDE.md: `ChatModelClient`/`EmbeddingClient` are never
merged), and the same reasoning applies to their recorders.

`EmbeddingCallRecord` has no field for the query text or the resulting vector, so neither can land
in the row by accident (same discipline as `ModelCallRecord`).
"""
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.types import EmbeddingKind
from app.db import repositories


@dataclass(frozen=True)
class EmbeddingCallRecord:
    kind: EmbeddingKind
    provider: str
    model_version: str
    batch_size: int
    latency_ms: int
    status: Literal["ok", "error"]
    error_code: str | None


class EmbeddingCallRecorder(Protocol):
    async def record(self, record: EmbeddingCallRecord) -> None: ...


@dataclass
class InMemoryEmbeddingCallRecorder:
    records: list[EmbeddingCallRecord] = field(default_factory=list)

    async def record(self, record: EmbeddingCallRecord) -> None:
        self.records.append(record)


class SqlTurnEmbeddingCallRecorder:
    """Writes to embedding_calls for one turn's query embeddings. A failed write propagates, same
    rule as `SqlModelCallRecorder`/`SqlToolCallRecorder`: silently losing the audit trail is worse
    than failing the request."""

    def __init__(self, engine: AsyncEngine, turn_id: UUID) -> None:
        self._engine = engine
        self._turn_id = turn_id

    async def record(self, record: EmbeddingCallRecord) -> None:
        await repositories.insert_embedding_call(self._engine, turn_id=self._turn_id, record=record)
