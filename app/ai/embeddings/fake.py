"""EmbeddingClient over the Week 2 deterministic embedder (wrapped, not rewritten).

Vectors carry no semantic meaning; identical text -> identical vector, which is enough for
plumbing tests and the `--fake` CLI. `fail_with` scripts failures: one exception fails every call,
a list fails calls in order and then succeeds.
"""
from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.embeddings.base import EmbeddingResponse
from app.ai.types import EmbeddingKind
from app.ingestion.fake_embedder import fake_embed


@dataclass(frozen=True)
class EmbedCall:
    texts: list[str]
    kind: EmbeddingKind
    model_version: str


class FakeEmbeddingClient:
    provider = "fake"

    def __init__(self, *, fail_with: BaseException | list[BaseException] | None = None) -> None:
        self._fail_with = fail_with
        self.calls: list[EmbedCall] = []

    async def embed(
        self, texts: Sequence[str], *, model_version: str, kind: EmbeddingKind
    ) -> EmbeddingResponse:
        self.calls.append(EmbedCall(list(texts), kind, model_version))
        if isinstance(self._fail_with, list):
            if self._fail_with:
                raise self._fail_with.pop(0)
        elif self._fail_with is not None:
            raise self._fail_with
        return EmbeddingResponse(fake_embed(list(texts)), self.provider, model_version, 0)
