import asyncio
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import Probe, get_probes
from app.main import create_app
from app.settings import Settings


async def ok() -> None:
    return None


async def boom() -> None:
    raise ConnectionError("postgresql://user:secret@db/x refused")


async def hang() -> None:
    await asyncio.sleep(10)


@pytest.fixture
def client_with(settings: Settings) -> Iterator[Callable[[dict[str, Probe]], TestClient]]:
    clients: list[TestClient] = []

    def make(probes: dict[str, Probe]) -> TestClient:
        app = create_app(settings)
        app.dependency_overrides[get_probes] = lambda: probes
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def test_healthz_touches_no_dependency(client_with: Callable[..., TestClient]) -> None:
    client = client_with({"postgres": boom, "rabbitmq": boom, "redis": boom})
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_all_ok(client_with: Callable[..., TestClient]) -> None:
    r = client_with({"postgres": ok, "rabbitmq": ok, "redis": ok}).get("/readyz")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "checks": {"postgres": "ok", "rabbitmq": "ok", "redis": "ok"},
    }


@pytest.mark.parametrize("down", ["postgres", "rabbitmq"])
def test_readyz_503_when_required_dependency_down(
    client_with: Callable[..., TestClient], down: str
) -> None:
    probes: dict[str, Probe] = {"postgres": ok, "rabbitmq": ok, "redis": ok}
    probes[down] = boom
    r = client_with(probes).get("/readyz")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"
    assert r.json()["checks"][down] == "fail"


def test_readyz_degraded_when_only_redis_down(client_with: Callable[..., TestClient]) -> None:
    r = client_with({"postgres": ok, "rabbitmq": ok, "redis": boom}).get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["checks"]["redis"] == "fail"


def test_readyz_hanging_probe_times_out_as_failure(
    client_with: Callable[..., TestClient],
) -> None:
    r = client_with({"postgres": hang, "rabbitmq": ok, "redis": ok}).get("/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["postgres"] == "fail"


def test_readyz_never_leaks_error_details(client_with: Callable[..., TestClient]) -> None:
    r = client_with({"postgres": boom, "rabbitmq": ok, "redis": ok}).get("/readyz")
    assert "secret" not in r.text
    assert "refused" not in r.text
