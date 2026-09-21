"""EmbeddingClient on Google's native SDK (gemini-embedding-001, 1536 dims).

* `output_dimensionality=EMBEDDING_DIM` — matches the VECTOR(1536) column; no migration.
* Vectors are L2-normalised here (`finalize_vectors`): Google normalises only at 3072 dims.
* No SDK retries (`attempts=1`): same reason as the chat adapter — a hidden retry is an uncounted
  call. Ingestion adds a *bounded, visible* backoff in SyncEmbedder instead.
"""
import time
from collections.abc import Sequence
from typing import Any

import httpx
import httpx2
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.ai.embeddings.base import EmbeddingResponse, finalize_vectors
from app.ai.errors import (
    EmbeddingInvalidError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from app.ai.types import EmbeddingKind
from app.db.models import EMBEDDING_DIM

_TASK_TYPE = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}
_TIMEOUTS = (httpx.TimeoutException, httpx2.TimeoutException)
_TRANSPORT = (httpx.TransportError, httpx2.TransportError)


def _retry_delay(exc: genai_errors.APIError) -> float | None:
    """Parse google.rpc.RetryInfo ("12s" / "12.5s") out of the error body, if present."""
    body: Any = exc.details
    try:
        for item in body["error"]["details"]:
            if str(item.get("@type", "")).endswith("RetryInfo"):
                return float(str(item["retryDelay"]).removesuffix("s"))
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return None


class GeminiEmbeddingClient:
    provider = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        http_async_client: httpx2.AsyncClient | None = None,
        timeout_seconds: float = 15.0,
        batch_size: int = 64,
    ) -> None:
        self._batch_size = batch_size
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                httpx_async_client=http_async_client,
                timeout=int(timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    async def embed(
        self, texts: Sequence[str], *, model_version: str, kind: EmbeddingKind
    ) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse([], self.provider, model_version, 0)
        started = time.perf_counter()
        raw: list[Sequence[float]] = []
        for start in range(0, len(texts), self._batch_size):
            raw.extend(await self._embed_batch(texts[start : start + self._batch_size],
                                               model_version, kind))
        vectors = finalize_vectors(raw, len(texts))
        latency_ms = int((time.perf_counter() - started) * 1000)
        return EmbeddingResponse(vectors, self.provider, model_version, latency_ms)

    async def _embed_batch(
        self, batch: Sequence[str], model: str, kind: EmbeddingKind
    ) -> list[Sequence[float]]:
        try:
            result = await self._client.aio.models.embed_content(
                model=model,
                contents=list(batch),
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIM, task_type=_TASK_TYPE[kind]
                ),
            )
        except genai_errors.APIError as exc:
            if exc.code == 429:
                raise ModelRateLimited(
                    "gemini embedding rate limited (HTTP 429)",
                    retry_after_seconds=_retry_delay(exc),
                ) from None
            raise ModelUnavailable(f"gemini embedding returned HTTP {exc.code}") from None
        except _TIMEOUTS:
            raise ModelTimeout("gemini embedding call timed out") from None
        except _TRANSPORT:
            raise ModelUnavailable("gemini embedding connection failed") from None

        vectors = [e.values for e in (result.embeddings or [])]
        if any(v is None for v in vectors):
            raise EmbeddingInvalidError("embedding entry without values")
        return vectors  # type: ignore[return-value]
