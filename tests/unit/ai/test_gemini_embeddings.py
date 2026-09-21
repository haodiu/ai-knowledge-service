"""GeminiEmbeddingClient against a mocked transport (google-genai over httpx2.MockTransport).

What this proves: the request carries output_dimensionality=1536 and the right task_type, the
adapter re-normalises (Google only normalises at 3072 dims), rejects wrong-sized vectors, does not
retry, and maps errors. What it CANNOT prove: that the live API honours output_dimensionality.
That is the `live` test in tests/live/.
"""
import json
import math

import httpx2
import pytest

from app.ai.embeddings.gemini import GeminiEmbeddingClient
from app.ai.errors import (
    EmbeddingDimensionError,
    EmbeddingInvalidError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from app.db.models import EMBEDDING_DIM

MODEL = "gemini-embedding-001"


class Server:
    def __init__(self, *, dim: int = EMBEDDING_DIM, value: float = 0.5, status: int = 200,
                 body: dict | None = None, raises: Exception | None = None) -> None:  # type: ignore[type-arg]
        self.requests: list[dict] = []  # type: ignore[type-arg]
        self.dim, self.value, self.status, self.body, self.raises = dim, value, status, body, raises

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        if self.raises:
            raise self.raises
        if self.status != 200:
            return httpx2.Response(self.status, json=self.body or {"error": {"message": "x"}})
        n = len(payload["requests"])
        return httpx2.Response(
            200, json={"embeddings": [{"values": [self.value] * self.dim} for _ in range(n)]}
        )

    def client(self, **kw: object) -> GeminiEmbeddingClient:
        return GeminiEmbeddingClient(
            api_key="test-key",
            http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self)),
            **kw,  # type: ignore[arg-type]
        )


async def test_dimension_matches_the_vector_column() -> None:
    """Locks 1536 for the real client too (mirror of the fake-embedder test from Week 2)."""
    resp = await Server().client().embed(["hello"], model_version=MODEL, kind="document")
    assert EMBEDDING_DIM == 1536
    assert len(resp.vectors) == 1 and len(resp.vectors[0]) == EMBEDDING_DIM


async def test_request_asks_for_1536_dims_and_the_matching_task_type() -> None:
    srv = Server()
    await srv.client().embed(["q"], model_version=MODEL, kind="query")
    await srv.client().embed(["d"], model_version=MODEL, kind="document")
    q, d = srv.requests
    assert q["requests"][0]["outputDimensionality"] == 1536
    assert q["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    assert d["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert q["requests"][0]["model"].endswith(MODEL)


async def test_vectors_are_l2_normalised_even_though_google_returns_them_unnormalised() -> None:
    resp = await Server(value=0.5).client().embed(["a", "b"], model_version=MODEL, kind="document")
    for vec in resp.vectors:
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("dim", [768, 3072, 1535])
async def test_wrong_sized_vectors_are_rejected_not_stored(dim: int) -> None:
    with pytest.raises(EmbeddingDimensionError):
        await Server(dim=dim).client().embed(["a"], model_version=MODEL, kind="document")


async def test_zero_vector_is_rejected() -> None:
    with pytest.raises(EmbeddingInvalidError):
        await Server(value=0.0).client().embed(["a"], model_version=MODEL, kind="document")


async def test_response_with_the_wrong_number_of_vectors_is_rejected() -> None:
    srv = Server(body={"embeddings": [{"values": [0.1] * EMBEDDING_DIM}]})
    srv.status = 200
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=srv.body)
    client = GeminiEmbeddingClient(
        api_key="k", http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )
    with pytest.raises(EmbeddingInvalidError):
        await client.embed(["a", "b"], model_version=MODEL, kind="document")


async def test_batches_are_split_and_order_is_preserved() -> None:
    srv = Server()
    texts = [f"t{i}" for i in range(130)]
    resp = await srv.client(batch_size=64).embed(texts, model_version=MODEL, kind="document")
    assert [len(r["requests"]) for r in srv.requests] == [64, 64, 2]
    sent = [r["content"]["parts"][0]["text"] for p in srv.requests for r in p["requests"]]
    assert sent == texts and len(resp.vectors) == 130


async def test_empty_input_makes_no_request() -> None:
    srv = Server()
    resp = await srv.client().embed([], model_version=MODEL, kind="document")
    assert resp.vectors == [] and srv.requests == []


async def test_429_is_rate_limited_and_is_never_retried() -> None:
    """A hidden SDK retry would bypass the bounded-call accounting (CLAUDE.md invariant #7)."""
    body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                      "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                   "retryDelay": "12s"}]}}
    srv = Server(status=429, body=body)
    with pytest.raises(ModelRateLimited) as exc:
        await srv.client().embed(["a"], model_version=MODEL, kind="query")
    assert len(srv.requests) == 1
    assert exc.value.retry_after_seconds == 12.0


async def test_429_without_retry_info_has_unknown_retry_after() -> None:
    srv = Server(status=429)
    with pytest.raises(ModelRateLimited) as exc:
        await srv.client().embed(["a"], model_version=MODEL, kind="query")
    assert exc.value.retry_after_seconds is None and len(srv.requests) == 1


@pytest.mark.parametrize("status", [500, 503])
async def test_server_errors_are_unavailable_and_not_retried(status: int) -> None:
    srv = Server(status=status)
    with pytest.raises(ModelUnavailable):
        await srv.client().embed(["a"], model_version=MODEL, kind="query")
    assert len(srv.requests) == 1


async def test_timeout_is_normalised() -> None:
    srv = Server(raises=httpx2.ReadTimeout("slow"))
    with pytest.raises(ModelTimeout):
        await srv.client().embed(["a"], model_version=MODEL, kind="query")
    assert len(srv.requests) == 1


async def test_api_key_is_not_leaked_into_error_messages() -> None:
    srv = Server(status=500)
    with pytest.raises(ModelUnavailable) as exc:
        await srv.client().embed(["a"], model_version=MODEL, kind="query")
    assert "test-key" not in str(exc.value)
