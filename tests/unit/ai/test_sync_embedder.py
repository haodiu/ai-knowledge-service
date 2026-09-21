"""SyncEmbedder: ingestion-only bridge (sync Embedder <- async EmbeddingClient) with bounded
backoff. Invariant #7 bounds the ONLINE graph; ingestion is idempotent (#10) so it may wait."""
from collections.abc import Iterator

import pytest

from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.embeddings.sync import SyncEmbedder
from app.ai.errors import ModelRateLimited, ModelUnavailable
from app.db.models import EMBEDDING_DIM

_made: list[SyncEmbedder] = []


@pytest.fixture(autouse=True)
def _close_embedders() -> Iterator[None]:
    yield
    while _made:
        _made.pop().close()  # each keeps a private event loop; don't leak it


def _embedder(client: FakeEmbeddingClient, sleeps: list[float], **kw: object) -> SyncEmbedder:
    emb = SyncEmbedder(client, model_version="m", sleep=sleeps.append, jitter=lambda: 0.0, **kw)  # type: ignore[arg-type]
    _made.append(emb)
    return emb


def test_returns_document_vectors_as_plain_lists() -> None:
    client = FakeEmbeddingClient()
    out = _embedder(client, [])(["a", "b"])
    assert len(out) == 2 and len(out[0]) == EMBEDDING_DIM
    assert client.calls[0].kind == "document"


def test_rate_limit_is_retried_honouring_retry_after_then_succeeds() -> None:
    client = FakeEmbeddingClient(
        fail_with=[ModelRateLimited("x", retry_after_seconds=2.0),
                   ModelRateLimited("x", retry_after_seconds=3.0)]
    )
    sleeps: list[float] = []
    assert len(_embedder(client, sleeps)(["a"])) == 1
    assert sleeps == [2.0, 3.0] and len(client.calls) == 3


def test_backoff_is_exponential_and_capped_when_retry_after_is_unknown() -> None:
    client = FakeEmbeddingClient(
        fail_with=[ModelRateLimited("x", retry_after_seconds=None)] * 6
    )
    sleeps: list[float] = []
    with pytest.raises(ModelRateLimited):
        _embedder(client, sleeps, max_attempts=6, base_delay=1.0, max_delay=8.0)(["a"])
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_gives_up_after_max_attempts() -> None:
    client = FakeEmbeddingClient(fail_with=[ModelRateLimited("x", retry_after_seconds=1.0)] * 9)
    with pytest.raises(ModelRateLimited):
        _embedder(client, [], max_attempts=3)(["a"])
    assert len(client.calls) == 3


def test_only_rate_limits_are_retried() -> None:
    client = FakeEmbeddingClient(fail_with=[ModelUnavailable("down")])
    with pytest.raises(ModelUnavailable):
        _embedder(client, [])(["a"])
    assert len(client.calls) == 1


async def test_refuses_to_run_inside_an_event_loop() -> None:
    with pytest.raises(RuntimeError, match="event loop"):
        _embedder(FakeEmbeddingClient(), [])(["a"])
