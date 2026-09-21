"""ONE OpenAI-compatible adapter (Gemini today; DeepSeek/MiniMax/GLM later), mocked transport."""
import json

import httpx2
import pytest

from app.ai.chat.openai_compat import OpenAICompatibleChatClient
from app.ai.errors import (
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
    StructuredOutputError,
)
from app.ai.schemas import QueryPlan
from app.ai.types import ModelMessage

BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MSGS = [ModelMessage(role="system", content="sys"), ModelMessage(role="user", content="hi")]
PLAN_JSON = json.dumps({"intent": "policy", "retrieval_query": "q", "needs_retrieval": True})


def _completion(content: str | None, finish: str = "stop") -> dict:  # type: ignore[type-arg]
    return {
        "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class Server:
    def __init__(self, *, status: int = 200, body: dict | None = None,  # type: ignore[type-arg]
                 headers: dict[str, str] | None = None, raises: Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.status, self.body, self.headers, self.raises = status, body, headers or {}, raises

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.raises:
            raise self.raises
        body = self.body if self.body is not None else _completion(PLAN_JSON)
        return httpx2.Response(self.status, json=body, headers=self.headers)

    def client(self, model: str = "gemini-3.1-flash-lite") -> OpenAICompatibleChatClient:
        return OpenAICompatibleChatClient(
            provider="gemini", base_url=BASE, api_key="test-key", model=model,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self)),
        )


async def _call(client: OpenAICompatibleChatClient) -> object:
    return await client.complete_structured(
        messages=MSGS, response_model=QueryPlan, purpose="plan", timeout_seconds=5
    )


async def test_happy_path_sends_json_schema_and_parses_into_the_model() -> None:
    srv = Server()
    client = srv.client("gemini-3.6-flash")
    resp = await client.complete_structured(
        messages=MSGS, response_model=QueryPlan, purpose="plan", timeout_seconds=5
    )
    (req,) = srv.requests
    body = json.loads(req.content)

    assert str(req.url) == BASE + "chat/completions"
    assert req.headers["authorization"] == "Bearer test-key"
    assert body["model"] == "gemini-3.6-flash"  # bare name, no "models/" prefix
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "QueryPlan"
    assert "retrieval_query" in rf["json_schema"]["schema"]["properties"]

    assert resp.parsed == QueryPlan.model_validate_json(PLAN_JSON)  # type: ignore[attr-defined]
    assert resp.provider == "gemini" and resp.model_name == "gemini-3.6-flash"  # type: ignore[attr-defined]
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (11, 7)  # type: ignore[attr-defined]
    assert client.provider == "gemini" and client.model_name == "gemini-3.6-flash"


@pytest.mark.parametrize(
    "content",
    ["not json", "", None, '{"intent":"policy"}',
     json.dumps({"intent": "hybrid", "retrieval_query": "q", "needs_retrieval": True,
                 "tool_request": {"name": "drop_table", "arguments": {"subscription_id": "s"}}})],
)
async def test_unusable_output_is_a_structured_output_error_and_not_retried(
    content: str | None,
) -> None:
    srv = Server(body=_completion(content))
    with pytest.raises(StructuredOutputError):
        await _call(srv.client())
    assert len(srv.requests) == 1


async def test_truncated_output_is_rejected_even_if_it_happens_to_parse() -> None:
    srv = Server(body=_completion(PLAN_JSON, finish="length"))
    with pytest.raises(StructuredOutputError):
        await _call(srv.client())


async def test_429_is_rate_limited_with_retry_after_and_sends_exactly_one_request() -> None:
    """The openai SDK defaults to max_retries=2 and would silently re-send: 3 requests for one
    logical call, invisible to model_calls and to MAX_GENERATIVE_LLM_CALLS (invariant #7)."""
    srv = Server(status=429, body={"error": {"message": "quota"}}, headers={"retry-after": "7"})
    with pytest.raises(ModelRateLimited) as exc:
        await _call(srv.client())
    assert len(srv.requests) == 1
    assert exc.value.retry_after_seconds == 7.0


async def test_429_without_header_has_unknown_retry_after() -> None:
    srv = Server(status=429, body={"error": {"message": "quota"}})
    with pytest.raises(ModelRateLimited) as exc:
        await _call(srv.client())
    assert exc.value.retry_after_seconds is None and len(srv.requests) == 1


@pytest.mark.parametrize("status", [500, 503, 401, 404])
async def test_other_http_errors_are_unavailable_and_sent_once(status: int) -> None:
    srv = Server(status=status, body={"error": {"message": "nope"}})
    with pytest.raises(ModelUnavailable):
        await _call(srv.client())
    assert len(srv.requests) == 1


async def test_timeout_is_normalised_and_not_retried() -> None:
    srv = Server(raises=httpx2.ReadTimeout("slow"))
    with pytest.raises(ModelTimeout):
        await _call(srv.client())
    assert len(srv.requests) == 1


async def test_connection_failure_is_unavailable_and_not_retried() -> None:
    srv = Server(raises=httpx2.ConnectError("down"))
    with pytest.raises(ModelUnavailable):
        await _call(srv.client())
    assert len(srv.requests) == 1


async def test_api_key_never_appears_in_error_messages() -> None:
    srv = Server(status=500, body={"error": {"message": "boom"}})
    with pytest.raises(ModelUnavailable) as exc:
        await _call(srv.client())
    assert "test-key" not in str(exc.value)
