"""Week 8, e2e scenario #11 (Plan §16.3): SSE emits only status events before validation, then
exactly one terminal event -- proven through the REAL HTTP endpoint, `run_turn()`, and a real
Postgres test DB, not the stubbed `run_turn` `tests/unit/api/test_conversations.py` uses to test
the streaming *wiring* in isolation. This is the gap that test explicitly documents as out of
scope for itself.

Models are fake (`build_fake_registry()`'s `demo_responder`, deterministic and schema-valid --
see `app.ai.chat.fake`), so this is about proving invariant #8 end to end, not model quality; a
live-model version belongs in `tests/live`, gated by `@pytest.mark.live`.

`get_engine` is deliberately NOT overridden: the app's real lifespan builds a real AsyncEngine from
`settings.database_url`, which points at the test DB (same approach as
`tests/integration/test_api_ingestions.py`), so the route's own DI resolves to it naturally and
`run_turn()` executes for real.
"""
import json
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, text

from app.ai.registry import build_fake_registry
from app.api.dependencies import enforce_rate_limit, get_auth, get_models
from app.auth.policies import AuthorizationContext
from app.main import create_app
from app.retrieval.schemas import Tier
from app.settings import Settings
from tests.integration.conftest import make_chunk, make_document, make_version, point_active

pytestmark = pytest.mark.integration

USER = AuthorizationContext(user_id="sse-e2e-user", tier=Tier.GENERAL)
REFUND_TEXT = (
    "Refunds are available within a 14 day refund window from the date of purchase. "
    "Contact support to start a refund."
)


@pytest.fixture
def client(test_db_url: str, db_engine: Engine) -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="test",
        database_url=SecretStr(test_db_url),
        redis_url=SecretStr("redis://localhost:1/0"),
        rabbitmq_url=SecretStr("amqp://u:p@localhost:1//"),
        jwt_secret=SecretStr("x"), jwt_issuer="x", jwt_audience="x",
        ingestion_service_token=SecretStr("x"),
    )
    app = create_app(settings)
    app.dependency_overrides[get_auth] = lambda: USER
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    app.dependency_overrides[get_models] = build_fake_registry
    # get_engine, get_prompts, get_tool_client are all left real: get_engine resolves to a real
    # AsyncEngine over test_db_url; get_prompts loads the real v1 prompts (no network);
    # get_tool_client degrades to None with no SUBSCRIPTION_SERVICE_BASE_URL/TOKEN configured
    # (app.api.dependencies' own documented behaviour), which this file's questions never need.
    with TestClient(app) as c:
        yield c


def _parse_sse(lines: list[str]) -> list[tuple[str, dict[str, object]]]:
    events = []
    event_name = None
    for line in lines:
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            assert event_name is not None
            events.append((event_name, json.loads(line.removeprefix("data: "))))
            event_name = None
    return events


def _seed_refund_chunk(db_engine: Engine) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    from app.ingestion.fake_embedder import fake_embed

    with db_engine.begin() as conn:
        doc = make_document(conn, "docs/refund.md", tier="general")
        version = make_version(conn, doc, 1, "active", content=REFUND_TEXT)
        point_active(conn, doc, version)
        chunk = make_chunk(conn, version, 0, REFUND_TEXT, fake_embed([REFUND_TEXT])[0])
    return doc, version, chunk


def test_real_turn_emits_only_status_events_then_one_validated_answer(
    client: TestClient, db_engine: Engine
) -> None:
    _doc, version, chunk = _seed_refund_chunk(db_engine)

    created = client.post("/v1/conversations", json={})
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    with client.stream(
        "POST",
        f"/v1/conversations/{conversation_id}/turns",
        json={"question": "What is the refund window?"},
    ) as r:
        turn_id = r.headers["x-turn-id"]
        events = _parse_sse(list(r.iter_lines()))

    names = [name for name, _ in events]
    assert names[:4] == ["status", "status", "status", "status"]
    assert names[-2:] == ["answer", "done"]
    # Everything between the status events and the terminal `answer` is answer_chunk (progressive
    # delivery of the already-validated text -- Plan §11.8 addendum).
    assert names[4:-2] == ["answer_chunk"] * len(names[4:-2])
    assert [data["phase"] for _, data in events[:4]] == [
        "planning", "retrieving", "grading", "generating",
    ]
    answer_event = events[-2][1]
    assert answer_event["answer"]  # demo_responder's deterministic answer text
    chunk_events = [data for name, data in events if name == "answer_chunk"]
    assert chunk_events  # demo_responder's answer is non-empty, so at least one chunk was sent
    assert "".join(str(c["delta"]) for c in chunk_events) == answer_event["answer"]
    citations = answer_event["citations"]
    assert isinstance(citations, list) and len(citations) == 1
    assert citations[0]["version_no"] == 1

    # DB side effects: the real graph actually ran, wrote model_calls, and persisted a citation
    # snapshot -- proving the SSE payload reflects real, validated evidence, not a stub.
    with db_engine.connect() as conn:
        purposes = sorted(
            r[0]
            for r in conn.execute(
                text("SELECT purpose FROM model_calls WHERE turn_id = :t"), {"t": turn_id}
            )
        )
        source = conn.execute(
            text(
                "SELECT document_version_id, chunk_id, text_snapshot "
                "FROM turn_sources WHERE turn_id = :t"
            ),
            {"t": turn_id},
        ).one()
        turn_row = conn.execute(
            text("SELECT graph_status, answer FROM turns WHERE id = :t"), {"t": turn_id}
        ).one()
    assert purposes == ["answer", "grade", "plan"]
    assert (source.document_version_id, source.chunk_id) == (version, chunk)
    assert source.text_snapshot == REFUND_TEXT
    assert turn_row.graph_status == "answered"
    assert turn_row.answer == answer_event["answer"]


def test_insufficient_evidence_still_only_emits_status_then_one_terminal_event(
    client: TestClient, db_engine: Engine
) -> None:
    """No seeded chunks at all: proves the "status events only, then exactly one terminal event"
    contract isn't specific to the `answered` case."""
    created = client.post("/v1/conversations", json={})
    conversation_id = created.json()["id"]

    with client.stream(
        "POST",
        f"/v1/conversations/{conversation_id}/turns",
        json={"question": "What is the refund window?"},
    ) as r:
        events = _parse_sse(list(r.iter_lines()))

    # route_after_retrieve short-circuits straight to the fallback when evidence is empty (Plan
    # §11.2: "nothing to grade: do not spend a call") -- no "grading" phase is emitted here, unlike
    # the answered case above.
    names = [name for name, _ in events]
    assert names == ["status", "status", "insufficient_evidence", "done"]
    assert [data["phase"] for _, data in events[:2]] == ["planning", "retrieving"]
