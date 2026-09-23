"""SSE conversation API (Plan §6, §11.8, invariant #8).

`run_turn()` and the repository calls are stubbed -- this tests the HTTP/streaming contract
(event ordering, auth, rate limiting, ownership), not the graph itself (covered elsewhere).
"""
import json
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import conversations as conv_module
from app.api.dependencies import (
    enforce_rate_limit,
    get_auth,
    get_engine,
    get_models,
    get_prompts,
    get_tool_client,
)
from app.auth.policies import AuthorizationContext
from app.db import repositories
from app.graph.result import TurnResult
from app.main import create_app
from app.retrieval.schemas import Tier
from app.settings import Settings
from app.tools.fake import FakeSubscriptionToolClient

USER = AuthorizationContext(user_id="user-1", tier=Tier.GENERAL)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[get_auth] = lambda: USER
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    app.dependency_overrides[get_engine] = lambda: object()
    app.dependency_overrides[get_models] = lambda: object()
    app.dependency_overrides[get_prompts] = lambda: object()
    app.dependency_overrides[get_tool_client] = lambda: FakeSubscriptionToolClient()
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


def _own(monkeypatch: pytest.MonkeyPatch, user_id: str | None) -> None:
    async def owner(engine: object, conversation_id: uuid.UUID) -> str | None:
        return user_id

    monkeypatch.setattr(repositories, "conversation_owner", owner)


def _stub_add_turn(monkeypatch: pytest.MonkeyPatch) -> uuid.UUID:
    turn_id = uuid.uuid4()

    async def add_turn(engine: object, conversation_id: uuid.UUID, question: str) -> uuid.UUID:
        return turn_id

    monkeypatch.setattr(repositories, "add_turn", add_turn)
    return turn_id


def test_missing_auth_header_is_401(client: TestClient) -> None:
    client.app.dependency_overrides.pop(get_auth)  # type: ignore[union-attr]
    r = client.post(f"/v1/conversations/{uuid.uuid4()}/turns", json={"question": "q"})
    assert r.status_code == 401


def test_posting_to_someone_elses_conversation_is_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _own(monkeypatch, "someone-else")
    r = client.post(f"/v1/conversations/{uuid.uuid4()}/turns", json={"question": "q"})
    assert r.status_code == 404


def test_posting_to_an_unknown_conversation_is_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _own(monkeypatch, None)
    r = client.post(f"/v1/conversations/{uuid.uuid4()}/turns", json={"question": "q"})
    assert r.status_code == 404


def test_rate_limit_exceeded_is_429(client: TestClient) -> None:
    async def deny() -> None:
        from fastapi import HTTPException

        raise HTTPException(429, detail="rate limit exceeded", headers={"Retry-After": "7"})

    client.app.dependency_overrides[enforce_rate_limit] = deny  # type: ignore[union-attr]
    r = client.post(f"/v1/conversations/{uuid.uuid4()}/turns", json={"question": "q"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"


def test_an_unconfigured_subscription_tool_does_not_crash_every_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: get_tool_client must return None, never raise, when
    SUBSCRIPTION_SERVICE_BASE_URL/TOKEN are unset (no Payment/Subscription host exists yet -- the
    `settings` fixture never sets them). Raising there would fail EVERY turn request via FastAPI's
    dependency resolution, before the route body or the graph ever runs -- including a plain
    'policy' question whose plan never proposes a tool. This removes the test client's usual
    get_tool_client override so the REAL dependency runs."""
    client.app.dependency_overrides.pop(get_tool_client)  # type: ignore[union-attr]
    _own(monkeypatch, USER.user_id)
    _stub_add_turn(monkeypatch)

    seen: dict[str, object] = {}

    async def fake_run_turn(**kwargs: object) -> TurnResult:
        seen["tool_client"] = kwargs["tool_client"]
        return TurnResult(status="answered", answer="Refunds within 14 days.", detail="ok")

    monkeypatch.setattr(conv_module, "run_turn", fake_run_turn)

    r = client.post(f"/v1/conversations/{uuid.uuid4()}/turns", json={"question": "q"})

    assert r.status_code == 200  # not a 500 from an unhandled ConfigurationError
    assert seen["tool_client"] is None  # the real get_tool_client, degrading gracefully


def test_sse_streams_status_events_then_one_answer_event(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _own(monkeypatch, USER.user_id)
    turn_id = _stub_add_turn(monkeypatch)

    async def fake_run_turn(**kwargs: object) -> TurnResult:
        on_phase = kwargs["on_phase"]
        for phase in ("planning", "retrieving", "grading", "generating"):
            await on_phase(phase)  # type: ignore[operator]
        return TurnResult(status="answered", answer="Refunds within 14 days.", detail="ok")

    monkeypatch.setattr(conv_module, "run_turn", fake_run_turn)

    with client.stream(
        "POST", f"/v1/conversations/{uuid.uuid4()}/turns", json={"question": "q"}
    ) as r:
        assert r.headers["x-turn-id"] == str(turn_id)  # the only way a client learns turn_id
        events = _parse_sse(list(r.iter_lines()))

    names = [name for name, _ in events]
    assert names == ["status", "status", "status", "status", "answer", "done"]
    assert [data["phase"] for _, data in events[:4]] == [
        "planning", "retrieving", "grading", "generating",
    ]
    assert events[4][1]["answer"] == "Refunds within 14 days."


def test_sse_never_emits_answer_before_status_events_finish(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generator cannot possibly emit `answer` before run_turn() returns -- there is no draft
    in scope until then (invariant #8). This proves the wiring reflects that, not just the type."""
    _own(monkeypatch, USER.user_id)
    _stub_add_turn(monkeypatch)
    order: list[str] = []

    async def fake_run_turn(**kwargs: object) -> TurnResult:
        on_phase = kwargs["on_phase"]
        await on_phase("planning")  # type: ignore[operator]
        order.append("phase-sent")
        return TurnResult(status="insufficient_evidence", detail="no_evidence")

    monkeypatch.setattr(conv_module, "run_turn", fake_run_turn)

    with client.stream(
        "POST", f"/v1/conversations/{uuid.uuid4()}/turns", json={"question": "q"}
    ) as r:
        events = _parse_sse(list(r.iter_lines()))

    assert order == ["phase-sent"]  # run_turn finished before the generator read the result
    assert [name for name, _ in events] == ["status", "insufficient_evidence", "done"]


def test_blocked_and_unavailable_get_their_own_event_names(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _own(monkeypatch, USER.user_id)
    _stub_add_turn(monkeypatch)

    async def fake_run_turn(**kwargs: object) -> TurnResult:
        return TurnResult(
            status="temporarily_unavailable", detail="rate_limited", retry_after_seconds=5.0
        )

    monkeypatch.setattr(conv_module, "run_turn", fake_run_turn)

    with client.stream(
        "POST", f"/v1/conversations/{uuid.uuid4()}/turns", json={"question": "q"}
    ) as r:
        events = _parse_sse(list(r.iter_lines()))

    assert events[0] == ("unavailable", {"detail": "rate_limited", "retry_after_seconds": 5.0})
