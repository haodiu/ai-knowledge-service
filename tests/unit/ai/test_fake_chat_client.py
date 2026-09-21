"""The fake ChatModelClient must be able to misbehave on purpose, through the REAL parse path.

If it only ever returned valid objects, none of the fail branches (bad JSON, wrong schema,
unknown tool, invented citation, provider errors) would be testable.
"""
import pytest

from app.ai.chat.fake import FakeChatModelClient
from app.ai.errors import ModelRateLimited, StructuredOutputError
from app.ai.schemas import EvidenceGrade, QueryPlan
from app.ai.types import ModelMessage

MSGS = [ModelMessage(role="system", content="s"), ModelMessage(role="user", content="u")]
PLAN = QueryPlan(intent="policy", retrieval_query="q", needs_retrieval=True)


async def _call(client: FakeChatModelClient, model=QueryPlan):  # type: ignore[no-untyped-def]
    return await client.complete_structured(
        messages=MSGS, response_model=model, purpose="plan", timeout_seconds=5
    )


async def test_scripted_valid_object_is_returned_parsed() -> None:
    client = FakeChatModelClient([PLAN])
    resp = await _call(client)
    assert resp.parsed == PLAN
    assert resp.provider == client.provider and resp.model_name == client.model_name
    assert resp.usage.input_tokens is not None


async def test_raw_json_text_and_dict_go_through_validation() -> None:
    client = FakeChatModelClient(
        ['{"intent":"policy","retrieval_query":"q","needs_retrieval":true}',
         {"intent": "policy", "retrieval_query": "q", "needs_retrieval": True}]
    )
    assert (await _call(client)).parsed == PLAN
    assert (await _call(client)).parsed == PLAN


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        "",
        '{"intent":"policy"',  # truncated
        {"intent": "policy", "retrieval_query": "q"},  # missing field
        {"intent": "policy", "retrieval_query": "q", "needs_retrieval": True,
         "tool_request": {"name": "drop_table", "arguments": {"subscription_id": "s"}}},
        {"intent": "policy", "retrieval_query": "x" * 501, "needs_retrieval": True},
    ],
)
async def test_malformed_output_raises_structured_output_error(bad: object) -> None:
    client = FakeChatModelClient([bad])  # type: ignore[list-item]
    with pytest.raises(StructuredOutputError):
        await _call(client)


async def test_structured_output_error_does_not_echo_the_model_output() -> None:
    client = FakeChatModelClient(["SECRET-ish model output {"])
    with pytest.raises(StructuredOutputError) as exc:
        await _call(client)
    assert "SECRET-ish" not in str(exc.value)


async def test_scripted_exception_is_raised() -> None:
    client = FakeChatModelClient([ModelRateLimited("slow down", retry_after_seconds=3.0)])
    with pytest.raises(ModelRateLimited) as exc:
        await _call(client)
    assert exc.value.retry_after_seconds == 3.0


async def test_script_is_consumed_in_order_and_exhaustion_is_loud() -> None:
    grade = EvidenceGrade(sufficient=True, confidence=0.9, reason="r")
    client = FakeChatModelClient([PLAN, grade])
    await _call(client)
    await _call(client, EvidenceGrade)
    with pytest.raises(AssertionError, match="more times than scripted"):
        await _call(client)


async def test_every_call_is_recorded_for_assertions() -> None:
    client = FakeChatModelClient([PLAN])
    await _call(client)
    (call,) = client.calls
    assert call.purpose == "plan" and call.messages == MSGS and call.response_model is QueryPlan


async def test_a_call_that_fails_is_still_recorded() -> None:
    client = FakeChatModelClient(["nope"])
    with pytest.raises(StructuredOutputError):
        await _call(client)
    assert len(client.calls) == 1
