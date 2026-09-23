"""Plan §16.3 #10: "Generator trả invalid structured output hai lần -> safe fallback, không phát
draft." See tests/e2e/__init__.py for what this directory is (and isn't) for Week 7.

The base case (policy-only turn) is already covered by
tests/unit/graph/test_failure_matrix.py's "invalid_output" entry. This file adds the Week 7
variant: a successful tool call and a successful retrieval both happened, evidence was graded
sufficient -- and the generator STILL fails twice. A good tool result must not be able to smuggle
an unvalidated answer past the citation gate (invariant #3, #8).
"""
import pytest

from app.ai.schemas import EvidenceGrade, QueryPlan
from app.retrieval.schemas import Evidence
from app.tools.fake import FakeSubscriptionToolClient
from app.tools.subscription import SubscriptionSnapshot
from tests.unit.ai.helpers import make_evidence
from tests.unit.graph.harness import Harness

pytestmark = pytest.mark.e2e

BAD = '{"status": "answered", "answer": "x"'  # truncated JSON, never valid


async def test_invalid_generator_output_is_safe_fallback_even_with_a_successful_tool_call() -> None:
    plan = QueryPlan.model_validate({
        "intent": "hybrid", "retrieval_query": "cancel policy", "needs_retrieval": True,
        "tool_request": {"name": "get_subscription", "arguments": {"subscription_id": "sub_9"}}})
    ev: list[Evidence] = make_evidence(1)
    grade = EvidenceGrade(sufficient=True, confidence=0.9, reason="covers it")
    snapshot = SubscriptionSnapshot(
        subscription_id="sub_9", customer_id="cus_1", status="active",
        observed_at="2026-09-23T00:00:00Z",
    )
    tool = FakeSubscriptionToolClient([snapshot])

    h = Harness([plan], [grade], [BAD, BAD], ev, tool=tool)
    result = await h.run()

    assert result.status == "blocked" and result.detail == "structured_output_invalid"
    assert result.tool_executed is True  # the tool DID run and DID succeed
    assert result.answer is None and result.citations == () and result.source_snapshots == ()
    assert "x" != result.answer  # the truncated draft text never leaks out
    assert [r.purpose for r in h.recorder.records][-2:] == ["answer", "repair"]
