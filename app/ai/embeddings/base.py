"""EmbeddingClient: the embedding seam (Plan §11.3). Deliberately NOT part of ChatModelClient —
different lifecycle (batched offline, one query online), provider, and failure behaviour."""
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.ai.errors import EmbeddingDimensionError, EmbeddingInvalidError
from app.ai.types import EmbeddingKind
from app.db.models import EMBEDDING_DIM


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: list[list[float]]  # len == len(texts); each EMBEDDING_DIM floats, L2-normalised
    provider: str
    model_version: str
    latency_ms: int


class EmbeddingClient(Protocol):
    # Mirrors ChatModelClient.provider (Plan §17): both real implementations already carry it;
    # declaring it here lets embed_query() record a provider on an error path too, before any
    # EmbeddingResponse exists to read one from.
    provider: str

    async def embed(
        self, texts: Sequence[str], *, model_version: str, kind: EmbeddingKind
    ) -> EmbeddingResponse:
        """Embed `texts`.

        Deliberate deviation from the Plan §11.3 signature: `kind` is added. Gemini's
        RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY are different task types, and using one for both
        degrades retrieval. Ingestion passes "document"; the online query path passes "query".
        """
        ...


def finalize_vectors(raw: Sequence[Sequence[float]], expected_count: int) -> list[list[float]]:
    """Validate count + dimension, then L2-normalise.

    Google only normalises at 3072 dims; at 1536 we must, to match the fake embedder and the
    cosine ops on chunks.embedding. A wrong-sized vector must never reach the VECTOR(1536) column.
    """
    if len(raw) != expected_count:
        raise EmbeddingInvalidError(f"expected {expected_count} vectors, got {len(raw)}")
    out: list[list[float]] = []
    for vec in raw:
        if len(vec) != EMBEDDING_DIM:
            raise EmbeddingDimensionError(f"expected {EMBEDDING_DIM} dims, got {len(vec)}")
        norm = math.sqrt(sum(x * x for x in vec))
        if not math.isfinite(norm) or norm == 0.0:
            raise EmbeddingInvalidError("embedding has zero or non-finite norm")
        out.append([x / norm for x in vec])
    return out
