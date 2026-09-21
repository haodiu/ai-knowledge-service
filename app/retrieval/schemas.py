import uuid
from dataclasses import dataclass
from enum import StrEnum


class Tier(StrEnum):
    """Mirrors the documents.tier CHECK constraint. Only ever supplied by trusted auth code."""

    GENERAL = "general"
    INTERNAL = "internal"


@dataclass(frozen=True)
class Evidence:
    """One retrieved chunk. (document_version_id, chunk_id) is the citation key (Plan §9.3)."""

    chunk_id: uuid.UUID
    document_version_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    version_no: int
    chunk_index: int
    text: str
    score: float
