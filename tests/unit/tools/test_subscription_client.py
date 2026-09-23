"""HttpSubscriptionToolClient against a mocked transport (Plan §10). Mirrors the shape of
tests/unit/ai/test_openai_compat_chat.py's Server/MockTransport pattern, but on plain `httpx`
(this client is not a provider-SDK adapter, so it does not use `httpx2`)."""
import httpx
import pytest

from app.ai.schemas import GetSubscriptionArgs
from app.tools.errors import (
    ToolAmbiguous,
    ToolNotFound,
    ToolRateLimited,
    ToolTimeout,
    ToolUnavailable,
)
from app.tools.subscription import HttpSubscriptionToolClient, SubscriptionSnapshot

BASE = "https://payment-platform.internal"


def _body(**overrides: object) -> dict[str, object]:
    base = {"subscription_id": "sub_1", "customer_id": "cus_1", "status": "active",
            "observed_at": "2026-09-23T00:00:00Z"}
    return {**base, **overrides}


class Server:
    def __init__(self, *, status: int = 200, body: dict[str, object] | None = None,
                 headers: dict[str, str] | None = None, raises: Exception | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.status, self.body, self.headers, self.raises = status, body, headers or {}, raises

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises:
            raise self.raises
        return httpx.Response(self.status, json=self.body if self.body is not None else _body(),
                              headers=self.headers)

    def client(self) -> HttpSubscriptionToolClient:
        return HttpSubscriptionToolClient(
            base_url=BASE, service_token="svc-token",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)),
        )


BY_SUB = GetSubscriptionArgs(subscription_id="sub_1")
BY_CUSTOMER = GetSubscriptionArgs(customer_id="cus_1")


async def test_happy_path_sends_service_token_and_on_behalf_of_header() -> None:
    srv = Server()
    resp = await srv.client().get_subscription(BY_SUB, user_id="user-1", timeout_seconds=5)
    (req,) = srv.requests

    assert str(req.url) == f"{BASE}/subscriptions?subscription_id=sub_1"
    assert req.headers["x-service-token"] == "svc-token"
    assert req.headers["x-on-behalf-of"] == "user-1"
    assert isinstance(resp, SubscriptionSnapshot)
    assert resp.subscription_id == "sub_1" and resp.status == "active"


async def test_customer_id_is_sent_when_subscription_id_is_absent() -> None:
    srv = Server()
    await srv.client().get_subscription(BY_CUSTOMER, user_id="u", timeout_seconds=5)
    (req,) = srv.requests
    assert str(req.url) == f"{BASE}/subscriptions?customer_id=cus_1"


async def test_extra_response_fields_are_ignored_not_rejected() -> None:
    """The host's body is third-party data, not one of our own contracts (extra="ignore")."""
    srv = Server(body=_body(plan_name="Pro", some_field_we_do_not_model="x"))
    resp = await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)
    assert resp.plan_name == "Pro"


async def test_404_is_not_found() -> None:
    srv = Server(status=404, body={"error": "not found"})
    with pytest.raises(ToolNotFound):
        await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)
    assert len(srv.requests) == 1


async def test_409_is_ambiguous() -> None:
    srv = Server(status=409, body={"error": "ambiguous"})
    with pytest.raises(ToolAmbiguous):
        await srv.client().get_subscription(BY_CUSTOMER, user_id="u", timeout_seconds=5)


async def test_429_is_rate_limited_with_retry_after() -> None:
    srv = Server(status=429, body={"error": "quota"}, headers={"retry-after": "12"})
    with pytest.raises(ToolRateLimited) as exc:
        await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)
    assert exc.value.retry_after_seconds == 12.0
    assert len(srv.requests) == 1  # no adapter-level retry, same rule as the chat adapter


async def test_429_without_header_has_unknown_retry_after() -> None:
    srv = Server(status=429, body={"error": "quota"})
    with pytest.raises(ToolRateLimited) as exc:
        await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)
    assert exc.value.retry_after_seconds is None


@pytest.mark.parametrize("status", [500, 503, 400])
async def test_other_http_statuses_are_unavailable(status: int) -> None:
    srv = Server(status=status, body={"error": "x"})
    with pytest.raises(ToolUnavailable):
        await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)


async def test_malformed_200_body_is_unavailable_not_a_crash() -> None:
    srv = Server(status=200, body={"status": "active"})  # missing required fields
    with pytest.raises(ToolUnavailable):
        await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)


async def test_timeout_is_normalised_and_sent_once() -> None:
    srv = Server(raises=httpx.ReadTimeout("slow"))
    with pytest.raises(ToolTimeout):
        await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)
    assert len(srv.requests) == 1


async def test_connection_failure_is_unavailable_and_sent_once() -> None:
    srv = Server(raises=httpx.ConnectError("down"))
    with pytest.raises(ToolUnavailable):
        await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)
    assert len(srv.requests) == 1


async def test_service_token_never_appears_in_error_messages() -> None:
    srv = Server(status=500, body={"error": "boom"})
    with pytest.raises(ToolUnavailable) as exc:
        await srv.client().get_subscription(BY_SUB, user_id="u", timeout_seconds=5)
    assert "svc-token" not in str(exc.value)
