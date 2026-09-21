"""model_calls recording: operational metadata only, by construction.

`ModelCallRecord` has no field for a prompt or a response, so raw text cannot be stored by
accident (Plan §5.4, §17, DoD). Recorders are handed a record, never messages.
"""
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.types import Purpose
from app.db import repositories


@dataclass(frozen=True)
class ModelCallRecord:
    purpose: Purpose
    provider: str
    model_name: str
    prompt_version: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    status: Literal["ok", "error"]
    error_code: str | None


class ModelCallRecorder(Protocol):
    async def record(self, record: ModelCallRecord) -> None: ...


@dataclass
class InMemoryModelCallRecorder:
    records: list[ModelCallRecord] = field(default_factory=list)

    async def record(self, record: ModelCallRecord) -> None:
        self.records.append(record)


class SqlModelCallRecorder:
    """Writes to model_calls for one turn. A failed write propagates: silently losing the audit
    trail is worse than failing the request."""

    def __init__(self, engine: AsyncEngine, turn_id: UUID) -> None:
        self._engine = engine
        self._turn_id = turn_id

    async def record(self, record: ModelCallRecord) -> None:
        await repositories.insert_model_call(self._engine, self._turn_id, record)
