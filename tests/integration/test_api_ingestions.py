"""Internal ingestion endpoint (Plan §6), deferred from Week 5. Celery dispatch is stubbed (the
task's own `.delay()`), not the DB layer -- `create_ingestion_job()`/`run_ingestion_job()` run for
real against the test database, same as the Week 5 CLI integration tests.
"""
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL, fake_embed
from app.ingestion.service import run_ingestion_job
from app.main import create_app
from app.settings import Settings

pytestmark = pytest.mark.integration

SERVICE_TOKEN = "test-service-token"


@pytest.fixture
def client(test_db_url: str, db_engine: object) -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="test",
        database_url=SecretStr(test_db_url),
        redis_url=SecretStr("redis://localhost:1/0"),
        rabbitmq_url=SecretStr("amqp://u:p@localhost:1//"),
        jwt_secret=SecretStr("x"), jwt_issuer="x", jwt_audience="x",
        ingestion_service_token=SecretStr(SERVICE_TOKEN),
    )
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(autouse=True)
def _stub_dispatch(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for the real `.delay()` publish -- no broker needed for this test file."""
    dispatched: list[str] = []

    class _Result:
        id = "fake-task-id"

    def fake_delay(job_id: str) -> _Result:
        dispatched.append(job_id)
        return _Result()

    from app.ingestion.tasks import ingest_document_task

    monkeypatch.setattr(ingest_document_task, "delay", fake_delay)
    return dispatched


def _body(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "external_id": "docs/refund.md", "title": "Refund policy", "tier": "general",
        "content": "Refunds within 14 days.", "embedding_model": FAKE_EMBEDDING_MODEL,
    }
    base.update(over)
    return base


def test_missing_service_token_is_401(client: TestClient) -> None:
    r = client.post("/internal/ingestions", json=_body())
    assert r.status_code == 401


def test_wrong_service_token_is_401(client: TestClient) -> None:
    r = client.post(
        "/internal/ingestions", json=_body(), headers={"X-Service-Token": "wrong"}
    )
    assert r.status_code == 401


def test_submit_creates_a_queued_job_and_dispatches_it(
    client: TestClient, _stub_dispatch: list[str]
) -> None:
    r = client.post(
        "/internal/ingestions", json=_body(), headers={"X-Service-Token": SERVICE_TOKEN}
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["document_external_id"] == "docs/refund.md"
    assert body["version_no"] == 1
    assert body["job_id"] is not None
    assert _stub_dispatch == [body["job_id"]]


def test_get_job_reflects_status_after_processing(
    client: TestClient, db_engine: object
) -> None:
    r = client.post(
        "/internal/ingestions", json=_body(), headers={"X-Service-Token": SERVICE_TOKEN}
    )
    job_id = r.json()["job_id"]

    r_before = client.get(
        f"/internal/ingestions/{job_id}", headers={"X-Service-Token": SERVICE_TOKEN}
    )
    assert r_before.json()["status"] == "queued"

    run_ingestion_job(db_engine, uuid.UUID(job_id), embed=fake_embed)  # type: ignore[arg-type]

    r_after = client.get(
        f"/internal/ingestions/{job_id}", headers={"X-Service-Token": SERVICE_TOKEN}
    )
    assert r_after.json()["status"] == "completed"


def test_get_unknown_job_is_404(client: TestClient) -> None:
    r = client.get(
        "/internal/ingestions/00000000-0000-0000-0000-000000000000",
        headers={"X-Service-Token": SERVICE_TOKEN},
    )
    assert r.status_code == 404


def test_resubmitting_superseded_content_still_returns_202_with_a_failed_job(
    client: TestClient, db_engine: object
) -> None:
    v1 = client.post(
        "/internal/ingestions", json=_body(content="V1"),
        headers={"X-Service-Token": SERVICE_TOKEN},
    ).json()
    run_ingestion_job(db_engine, uuid.UUID(v1["job_id"]), embed=fake_embed)  # type: ignore[arg-type]
    v2 = client.post(
        "/internal/ingestions", json=_body(content="V2"),
        headers={"X-Service-Token": SERVICE_TOKEN},
    ).json()
    run_ingestion_job(db_engine, uuid.UUID(v2["job_id"]), embed=fake_embed)  # type: ignore[arg-type]

    r = client.post(
        "/internal/ingestions", json=_body(content="V1"),
        headers={"X-Service-Token": SERVICE_TOKEN},
    )

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "failed"
    assert body["job_id"] is not None
    r_job = client.get(
        f"/internal/ingestions/{body['job_id']}", headers={"X-Service-Token": SERVICE_TOKEN}
    )
    assert r_job.json()["error_code"] == "superseded_content"
