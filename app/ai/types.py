"""Value types shared by the chat and embedding sides (kept protocol-free on purpose)."""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

# Subset of model_calls.purpose (Plan §5.4); 'repair' is reserved for Week 4's bounded repair.
Purpose = Literal["plan", "grade", "answer"]
EmbeddingKind = Literal["document", "query"]


@dataclass(frozen=True)
class ModelMessage:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class ModelResponse[T: BaseModel]:
    parsed: T  # already validated; a raw provider string is never returned to callers
    provider: str
    model_name: str
    usage: Usage
    latency_ms: int
