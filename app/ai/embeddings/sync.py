"""Sync `Embedder` (what ingestion.service expects) over an async EmbeddingClient.

INGESTION ONLY. It may back off and retry rate limits because ingestion is idempotent (invariant
#10) and embeds outside any DB transaction; invariant #7's bounds apply to the online graph, which
must never use this class. The waiting is bounded (max_attempts, max_delay) and visible.

It keeps ONE private event loop for its lifetime: SDK connection pools are bound to the loop that
first used them, so a fresh `asyncio.run` per call would break the second call.
"""
import asyncio
import random
import time
from collections.abc import Callable

from app.ai.embeddings.base import EmbeddingClient
from app.ai.errors import ModelRateLimited
from app.ai.types import EmbeddingKind


class SyncEmbedder:
    def __init__(
        self,
        client: EmbeddingClient,
        *,
        model_version: str,
        kind: EmbeddingKind = "document",
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.0, 0.25),
    ) -> None:
        self._client = client
        self._model_version = model_version
        self._kind: EmbeddingKind = kind
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter
        self._loop: asyncio.AbstractEventLoop | None = None

    def __call__(self, texts: list[str]) -> list[list[float]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("SyncEmbedder cannot run inside a running event loop")
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self._loop.run_until_complete(
                    self._client.embed(texts, model_version=self._model_version, kind=self._kind)
                )
                return resp.vectors
            except ModelRateLimited as exc:
                if attempt == self._max_attempts:
                    raise
                delay = exc.retry_after_seconds
                if delay is None:
                    delay = self._base_delay * 2 ** (attempt - 1)
                self._sleep(min(delay, self._max_delay) + self._jitter())
        raise AssertionError("unreachable")  # pragma: no cover

    def close(self) -> None:
        if self._loop is not None:
            self._loop.close()
            self._loop = None
