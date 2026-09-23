"""tool_node (app/graph/nodes/tools.py): exercised directly so its returned state dict and the
tool_calls recorder rows can be asserted precisely, one outcome at a time."""
from app.ai.schemas import QueryPlan
from app.graph.nodes.tools import tool_node
from app.tools.errors import (
    ToolAmbiguous,
    ToolNotFound,
    ToolRateLimited,
    ToolTimeout,
    ToolUnavailable,
)
from app.tools.fake import FakeSubscriptionToolClient
from app.tools.subscription import SubscriptionSnapshot
from tests.unit.graph.harness import Harness

PLAN = QueryPlan.model_validate({
    "intent": "subscription", "retrieval_query": "x", "needs_retrieval": False,
    "tool_request": {"name": "get_subscription", "arguments": {"customer_id": "c1"}}})
SNAPSHOT = SubscriptionSnapshot(
    subscription_id="s1", customer_id="c1", status="active",
    observed_at="2026-09-23T00:00:00Z",
)


async def test_success_adds_evidence_and_records_ok() -> None:
    h = Harness(tool=FakeSubscriptionToolClient([SNAPSHOT]))
    out = await tool_node({"plan": PLAN, "evidence": []}, h.ctx())

    assert out["tool_executed"] is True
    assert "error" not in out
    assert len(out["evidence"]) == 1 and out["evidence"][0].title == "Subscription s1"
    (record,) = h.tool_recorder.records
    assert (record.tool_name, record.identifier_kind, record.status, record.error_code) == \
        ("get_subscription", "customer_id", "ok", None)


async def test_404_returns_no_error_and_no_evidence() -> None:
    """Invariant #4: no error state at all -- falls through to the ordinary no_evidence path."""
    h = Harness(tool=FakeSubscriptionToolClient([ToolNotFound()]))
    out = await tool_node({"plan": PLAN, "evidence": []}, h.ctx())

    assert out == {"tool_executed": True}
    assert h.tool_recorder.records[0].status == "error"
    assert h.tool_recorder.records[0].error_code == "tool_not_found"


async def test_409_is_a_clarification_error_with_a_question() -> None:
    h = Harness(tool=FakeSubscriptionToolClient([ToolAmbiguous("more than one match")]))
    out = await tool_node({"plan": PLAN, "evidence": []}, h.ctx())

    assert out["tool_executed"] is True
    assert out["error"].kind == "clarification" and out["error"].code == "tool_ambiguous"
    assert out["error"].clarification_question


async def test_timeout_is_unavailable() -> None:
    h = Harness(tool=FakeSubscriptionToolClient([ToolTimeout()]))
    out = await tool_node({"plan": PLAN, "evidence": []}, h.ctx())
    assert out["error"].kind == "unavailable" and out["error"].code == "tool_timeout"


async def test_rate_limited_carries_retry_after_seconds() -> None:
    h = Harness(tool=FakeSubscriptionToolClient([ToolRateLimited(retry_after_seconds=9.0)]))
    out = await tool_node({"plan": PLAN, "evidence": []}, h.ctx())
    assert out["error"].code == "tool_rate_limited" and out["error"].retry_after_seconds == 9.0


async def test_5xx_is_unavailable() -> None:
    h = Harness(tool=FakeSubscriptionToolClient([ToolUnavailable("503")]))
    out = await tool_node({"plan": PLAN, "evidence": []}, h.ctx())
    assert out["error"].code == "tool_unavailable"


async def test_no_tool_client_configured_is_a_turn_scoped_unavailable_not_a_crash() -> None:
    """Regression: app/api/dependencies.py::get_tool_client returns None (never raises) when no
    Payment/Subscription host is configured, so this must be a normal, named turn outcome -- not
    an unhandled exception that would take down every request, including ones whose plan never
    proposes a tool at all."""
    h = Harness(tool=None)
    out = await tool_node({"plan": PLAN, "evidence": []}, h.ctx())

    assert out["tool_executed"] is True
    assert out["error"].kind == "unavailable" and out["error"].code == "tool_unavailable"
    assert h.tool_recorder.records == []  # no attempt was made, so no row


async def test_budget_exhaustion_makes_zero_outbound_calls_and_zero_recorder_rows() -> None:
    """Mirrors test_a_call_that_cannot_get_a_unit_ends_the_turn_before_any_request (chat side):
    acquire_tool() must raise BEFORE any request, so a MAX_TOOL_CALLS bypass in routing can never
    actually reach the host."""
    tool = FakeSubscriptionToolClient([SNAPSHOT])
    h = Harness(tool=tool)
    h.budget.max_tool_calls = 0
    out = await tool_node({"plan": PLAN, "evidence": []}, h.ctx())

    assert out["tool_executed"] is True
    assert out["error"].code == "budget_exhausted"
    assert tool.calls == []
    assert h.tool_recorder.records == []
