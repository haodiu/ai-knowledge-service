"""The subscription tool's contract with the host Spring Boot service (Plan §10, §16.2).

No such host exists yet (see CLAUDE.md/Plan §18 Tuần 7) -- this is the "mock Spring Boot service"
the plan's own testing strategy calls for, standing in for it via `httpx.MockTransport`. It is
kept separate from `tests/unit/tools/test_subscription_client.py` (client-internals unit test) so
a reviewer can point at exactly this file as "the promised contract test," and so it survives as
the thing to re-run against the real host's fixtures once one exists (see CLAUDE.md invariant
about not weakening/repurposing a test silently -- when the host is real, this file's `MockHost`
gets replaced by a live call, the request/response assertions stay).
"""
import httpx
import pytest

from app.ai.schemas import GetSubscriptionArgs
from app.tools.errors import ToolAmbiguous, ToolNotFound, ToolRateLimited, ToolUnavailable
from app.tools.subscription import HttpSubscriptionToolClient, SubscriptionSnapshot

pytestmark = pytest.mark.contract

BASE = "https://payment-platform.internal"
SERVICE_TOKEN = "svc-token"  # noqa: S105 -- test fixture value, not a real credential
USER_ID = "user-42"

# Plan §10's response table, reproduced as a fixture: (host status, host body) -> outcome.
_HOST_RESPONSE_TABLE = {
    "200_found": (200, {"subscription_id": "sub_1", "customer_id": "cus_1", "status": "active",
                        "plan_name": "Pro", "observed_at": "2026-09-23T00:00:00Z"}),
    "404_not_found": (404, {"error": "not_found"}),
    "404_not_authorized": (404, {"error": "forbidden"}),  # same status as not_found -- invariant #4
    "409_ambiguous": (409, {"error": "ambiguous_identifier"}),
    "429_rate_limited": (429, {"error": "rate_limited"}),
    "5xx_down": (503, {"error": "service_unavailable"}),
}


class MockHost:
    """Stands in for the not-yet-existing Payment/Subscription host, scripted with one entry from
    `_HOST_RESPONSE_TABLE`."""

    def __init__(self, outcome: str, *, retry_after: str | None = None) -> None:
        self.status, self.body = _HOST_RESPONSE_TABLE[outcome]
        self.headers = {"retry-after": retry_after} if retry_after else {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body, headers=self.headers)

    def client(self) -> HttpSubscriptionToolClient:
        return HttpSubscriptionToolClient(
            base_url=BASE, service_token=SERVICE_TOKEN,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)),
        )


async def test_request_shape_carries_service_credential_and_on_behalf_of_identity() -> None:
    host = MockHost("200_found")
    await host.client().get_subscription(
        GetSubscriptionArgs(subscription_id="sub_1"), user_id=USER_ID, timeout_seconds=5,
    )
    (req,) = host.requests
    assert req.method == "GET"
    assert req.url.path == "/subscriptions"
    assert dict(httpx.QueryParams(req.url.query)) == {"subscription_id": "sub_1"}
    assert req.headers["x-service-token"] == SERVICE_TOKEN
    assert req.headers["x-on-behalf-of"] == USER_ID
    # the raw JWT never reaches this layer (invariant #1) -- only a bare user_id string does
    assert "bearer" not in {k.lower(): v for k, v in req.headers.items()}.get("authorization", "")


async def test_200_round_trips_into_a_typed_snapshot() -> None:
    host = MockHost("200_found")
    snapshot = await host.client().get_subscription(
        GetSubscriptionArgs(subscription_id="sub_1"), user_id=USER_ID, timeout_seconds=5,
    )
    assert isinstance(snapshot, SubscriptionSnapshot)
    assert (snapshot.subscription_id, snapshot.status) == ("sub_1", "active")


@pytest.mark.parametrize("outcome", ["404_not_found", "404_not_authorized"])
async def test_404_is_tool_not_found_regardless_of_the_real_reason(outcome: str) -> None:
    """Invariant #4: the client (and everything above it) cannot tell these apart -- the host
    collapsing them into the same status code is the point, not a gap this client should try to
    work around."""
    host = MockHost(outcome)
    with pytest.raises(ToolNotFound):
        await host.client().get_subscription(
            GetSubscriptionArgs(customer_id="cus_1"), user_id=USER_ID, timeout_seconds=5,
        )


async def test_409_is_tool_ambiguous() -> None:
    host = MockHost("409_ambiguous")
    with pytest.raises(ToolAmbiguous):
        await host.client().get_subscription(
            GetSubscriptionArgs(customer_id="cus_1"), user_id=USER_ID, timeout_seconds=5,
        )


async def test_429_carries_the_hosts_retry_after() -> None:
    host = MockHost("429_rate_limited", retry_after="15")
    with pytest.raises(ToolRateLimited) as exc:
        await host.client().get_subscription(
            GetSubscriptionArgs(subscription_id="sub_1"), user_id=USER_ID, timeout_seconds=5,
        )
    assert exc.value.retry_after_seconds == 15.0


async def test_host_down_is_temporarily_unavailable() -> None:
    host = MockHost("5xx_down")
    with pytest.raises(ToolUnavailable):
        await host.client().get_subscription(
            GetSubscriptionArgs(subscription_id="sub_1"), user_id=USER_ID, timeout_seconds=5,
        )


async def test_the_configured_timeout_is_forwarded_to_the_transport() -> None:
    """A short, non-configurable-by-the-model timeout (Plan §10 "timeout ngắn") -- proven by
    reading back what httpx actually attached to the outbound request, not by re-testing the
    ReadTimeout -> ToolTimeout mapping (already covered in test_subscription_client.py)."""
    host = MockHost("200_found")
    await host.client().get_subscription(
        GetSubscriptionArgs(subscription_id="sub_1"), user_id=USER_ID, timeout_seconds=2.5,
    )
    (req,) = host.requests
    timeout = req.extensions["timeout"]
    assert timeout == {"connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5}
