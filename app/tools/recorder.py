"""tool_calls recording: operational metadata only, by construction -- mirrors
`app/ai/recorder.py`. Kept as a sibling module, not folded into it: tool calls and model calls are
a different-lifecycle pair (host HTTP vs. LLM provider), the same reasoning CLAUDE.md gives for
never merging `ChatModelClient`/`EmbeddingClient`.

`ToolCallRecord` has no field for the raw `subscription_id`/`customer_id` value, only which kind of
identifier was used -- so a PII-shaped value cannot land in the row by accident.
"""
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine

from app.db import repositories

IdentifierKind = Literal["subscription_id", "customer_id"]


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    identifier_kind: IdentifierKind
    latency_ms: int
    status: Literal["ok", "error"]
    error_code: str | None


class ToolCallRecorder(Protocol):
    async def record(self, record: ToolCallRecord) -> None: ...


@dataclass
class InMemoryToolCallRecorder:
    records: list[ToolCallRecord] = field(default_factory=list)

    async def record(self, record: ToolCallRecord) -> None:
        self.records.append(record)


class SqlToolCallRecorder:
    """Writes to tool_calls for one turn. A failed write propagates, same rule as
    `SqlModelCallRecorder`: silently losing the audit trail is worse than failing the request."""

    def __init__(self, engine: AsyncEngine, turn_id: UUID) -> None:
        self._engine = engine
        self._turn_id = turn_id

    async def record(self, record: ToolCallRecord) -> None:
        await repositories.insert_tool_call(self._engine, self._turn_id, record)
