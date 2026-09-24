"""Correlation-id middleware (Plan §17, Week 8). See app/api/middleware.py."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import RequestIdMiddleware
from app.logging_setup import current_request_id


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/probe")
    def probe() -> dict[str, str | None]:
        return {"seen": current_request_id()}

    return app


def test_generates_a_request_id_when_none_is_provided() -> None:
    client = TestClient(_app())

    r = client.get("/probe")

    assert r.status_code == 200
    header_id = r.headers["x-request-id"]
    assert header_id  # non-empty
    assert r.json()["seen"] == header_id  # the same id was visible to the route handler


def test_echoes_a_provided_request_id_unchanged() -> None:
    client = TestClient(_app())

    r = client.get("/probe", headers={"X-Request-Id": "caller-supplied-id"})

    assert r.headers["x-request-id"] == "caller-supplied-id"
    assert r.json()["seen"] == "caller-supplied-id"


def test_no_leakage_across_sequential_requests() -> None:
    client = TestClient(_app())

    first = client.get("/probe", headers={"X-Request-Id": "first"})
    second = client.get("/probe")  # no header this time -- must not see "first"

    assert first.json()["seen"] == "first"
    assert second.json()["seen"] != "first"
    assert current_request_id() is None  # cleared outside any request too
