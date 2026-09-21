"""Deterministic, hash-based embedder for Week 2 (no provider exists until Week 3).

The vectors carry NO semantic meaning: identical text -> identical vector, that is all. Vector
ranking on data ingested with this is meaningless; only full-text search is meaningful in the
Week 2 CLI demo. Real embeddings come from Week 3's EmbeddingClient.
"""
import hashlib
import math

from app.db.models import EMBEDDING_DIM

FAKE_EMBEDDING_MODEL = f"fake-shake256-{EMBEDDING_DIM}"


def _embed_one(text: str) -> list[float]:
    raw = hashlib.shake_256(text.encode("utf-8")).digest(EMBEDDING_DIM * 2)
    # 2 bytes per dimension, mapped to [-1, 1), then L2-normalised (cosine ops expect direction).
    vec = [int.from_bytes(raw[i : i + 2], "big") / 32768.0 - 1.0 for i in range(0, len(raw), 2)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [_embed_one(t) for t in texts]
