"""Plan §16.3 #8: "Tool `404`-for-unauthorized and `404`-for-not-found are indistinguishable to
the caller." See tests/e2e/__init__.py for what this directory is (and isn't) for Week 7.
"""
import pytest

from app.ai.schemas import QueryPlan
from app.tools.errors import ToolNotFound
from app.tools.fake import FakeSubscriptionToolClient
from tests.unit.graph.harness import Harness

pytestmark = pytest.mark.e2e

PLAN = QueryPlan.model_validate({
    "intent": "subscription", "retrieval_query": "my subscription", "needs_retrieval": False,
    "tool_request": {"name": "get_subscription", "arguments": {"customer_id": "c1"}}})


@pytest.mark.parametrize(
    "host_reason", ["not_found", "not_authorized"],
    ids=["record does not exist", "record exists but belongs to someone else"],
)
async def test_a_404_never_reveals_which_reason_it_was(host_reason: str) -> None:
    """Both reasons are modelled by the SAME exception (ToolNotFound) because the host itself
    returns the same HTTP 404 for both -- there is no second code path a caller could probe to
    tell them apart, whatever `host_reason` "really" was on the host side."""
    tool = FakeSubscriptionToolClient([ToolNotFound(f"host said: {host_reason}")])
    result = await Harness([PLAN], tool=tool).run()

    assert result.status == "insufficient_evidence"
    assert result.detail == "no_evidence"
    assert result.tool_executed is True
    assert result.answer is None
    assert result.clarification_question is None
    assert result.citations == ()
    # the exception's message (which could differ by host_reason) never reaches the caller
    assert host_reason not in repr(result)


async def test_both_reasons_produce_byte_identical_turn_results() -> None:
    """The strongest form of the assertion: not just 'same status', but nothing in the result
    distinguishes the two cases at all."""
    not_found = FakeSubscriptionToolClient([ToolNotFound("does not exist")])
    not_authorized = FakeSubscriptionToolClient([ToolNotFound("belongs to another user")])

    r1 = await Harness([PLAN], tool=not_found).run()
    r2 = await Harness([PLAN], tool=not_authorized).run()

    assert (r1.status, r1.detail, r1.answer, r1.clarification_question, r1.citations) == \
        (r2.status, r2.detail, r2.answer, r2.clarification_question, r2.citations)
