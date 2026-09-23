"""Scriptable fake SubscriptionToolClient -- able to misbehave on purpose.

Same shape as `app/ai/chat/fake.py::FakeChatModelClient`: a script of return values / exceptions,
consumed in order; running past the script is an AssertionError, which is how a test proves "the
tool was never called" (Plan §16.3 #9: an unknown/rejected tool proposal must produce zero
outbound calls). Used by tests and by the CLI's `--fake` mode.
"""
from collections.abc import Iterable
from dataclasses import dataclass

from app.ai.schemas import GetSubscriptionArgs
from app.tools.subscription import SubscriptionSnapshot

Scripted = SubscriptionSnapshot | BaseException


@dataclass(frozen=True)
class FakeToolCall:
    args: GetSubscriptionArgs
    user_id: str
    timeout_seconds: float


class FakeSubscriptionToolClient:
    def __init__(self, script: Iterable[Scripted] = ()) -> None:
        self._script = list(script)
        self.calls: list[FakeToolCall] = []

    async def get_subscription(
        self, args: GetSubscriptionArgs, *, user_id: str, timeout_seconds: float
    ) -> SubscriptionSnapshot:
        self.calls.append(FakeToolCall(args, user_id, timeout_seconds))
        if not self._script:
            raise AssertionError("fake subscription tool client called more times than scripted")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
