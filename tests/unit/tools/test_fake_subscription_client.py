"""FakeSubscriptionToolClient: same scripted-client contract as FakeChatModelClient
(tests/unit/ai/test_fake_chat_client.py), so graph tests can prove "the tool was never called"."""
import pytest

from app.ai.schemas import GetSubscriptionArgs
from app.tools.errors import ToolNotFound
from app.tools.fake import FakeSubscriptionToolClient
from app.tools.subscription import SubscriptionSnapshot

ARGS = GetSubscriptionArgs(subscription_id="sub_1")
SNAPSHOT = SubscriptionSnapshot(
    subscription_id="sub_1", customer_id="cus_1", status="active",
    observed_at="2026-09-23T00:00:00Z",
)


async def test_returns_scripted_items_in_order() -> None:
    client = FakeSubscriptionToolClient([SNAPSHOT])
    result = await client.get_subscription(ARGS, user_id="u1", timeout_seconds=5)
    assert result is SNAPSHOT
    assert len(client.calls) == 1
    assert client.calls[0].args is ARGS and client.calls[0].user_id == "u1"


async def test_a_scripted_exception_is_raised() -> None:
    client = FakeSubscriptionToolClient([ToolNotFound("nope")])
    with pytest.raises(ToolNotFound):
        await client.get_subscription(ARGS, user_id="u1", timeout_seconds=5)
    assert len(client.calls) == 1  # the call still counts even though it raised


async def test_calling_past_the_script_is_an_assertion_error() -> None:
    """Plan §16.3 #9: this is how a test proves an unauthorized/rejected proposal never reached
    the tool -- calling an empty-scripted fake raises instead of silently returning something."""
    client = FakeSubscriptionToolClient([])
    with pytest.raises(AssertionError):
        await client.get_subscription(ARGS, user_id="u1", timeout_seconds=5)


async def test_empty_client_has_zero_calls_until_used() -> None:
    client = FakeSubscriptionToolClient()
    assert client.calls == []
