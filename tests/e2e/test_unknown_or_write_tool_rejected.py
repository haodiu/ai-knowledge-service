"""Plan §16.3 #9: "Planner đề xuất unknown/write tool -> application reject và không có outbound
call." See tests/e2e/__init__.py for what this directory is (and isn't) for Week 7.

`ToolName = Literal["get_subscription"]` (app/ai/schemas.py) is the whole allowlist, and there is
no write tool in this codebase at all (CLAUDE.md invariant #11) -- so both "unknown" and "write"
collapse to the same case: the planner's structured output naming anything else fails Pydantic
validation before a `QueryPlan` can exist, which is a repaired `StructuredOutputError`
(invariant #7), never a tool call.
"""
import pytest

from tests.unit.graph.harness import Harness

pytestmark = pytest.mark.e2e

# A "write" tool proposal, dressed up as JSON the planner might emit if it ignored the allowlist.
_UNKNOWN_TOOL = (
    '{"intent": "subscription", "retrieval_query": "cancel my plan", "needs_retrieval": false, '
    '"tool_request": {"name": "cancel_subscription", "arguments": {"customer_id": "c1"}}}'
)
_WRITE_TOOL = (
    '{"intent": "subscription", "retrieval_query": "delete my account", "needs_retrieval": '
    'false, "tool_request": {"name": "delete_account", "arguments": {"customer_id": "c1"}}}'
)


@pytest.mark.parametrize("bad_proposal", [_UNKNOWN_TOOL, _WRITE_TOOL])
async def test_a_disallowed_tool_proposal_makes_zero_outbound_tool_calls(bad_proposal: str) -> None:
    # scripted twice: the one repair attempt (invariant #7) also fails, since the model repeats
    # the same disallowed name -- proving the tool client is unreachable even after a retry.
    h = Harness([bad_proposal, bad_proposal])
    result = await h.run()

    assert h.tool_calls == 0
    assert result.status == "blocked" and result.detail == "structured_output_invalid"
    assert result.proposed_tool is None  # no QueryPlan naming the bad tool ever existed
    assert len(h.planner.calls) == 2  # rejected attempt + the one repair, no more
