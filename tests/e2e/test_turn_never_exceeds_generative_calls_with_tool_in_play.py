"""Plan §16.3 #12: "Một turn không vượt MAX_GENERATIVE_LLM_CALLS kể cả khi tool tham gia."
See tests/e2e/__init__.py for what this directory is (and isn't) for Week 7.

The tool call draws from its OWN budget (`TurnBudget.acquire_tool`, `MAX_TOOL_CALLS=1`), entirely
separate from the generative-call pool (`MAX_GENERATIVE_LLM_CALLS=4`) that `app/ai/roles.py::_call`
guards -- this asserts both bounds hold independently in the SAME turn, not just each in isolation
(each is already fuzz-tested alone: tests/unit/graph/test_workflow.py's
`test_no_failure_sequence_can_exceed_the_hard_limits`).
"""
import pytest

from app.ai.errors import ModelUnavailable
from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from app.graph.limits import MAX_GENERATIVE_LLM_CALLS, MAX_TOOL_CALLS
from app.retrieval.schemas import Evidence
from app.tools.fake import FakeSubscriptionToolClient
from app.tools.subscription import SubscriptionSnapshot
from tests.unit.ai.helpers import make_evidence
from tests.unit.graph.harness import Harness

pytestmark = pytest.mark.e2e


def _answered(ev: Evidence) -> AnswerDraft:
    return AnswerDraft(status="answered", answer="14 days.", citations=[
        CitationRef(document_version_id=ev.document_version_id, chunk_id=ev.chunk_id)])


async def test_a_transient_grader_retry_plus_a_successful_tool_call_stays_within_both_bounds() \
        -> None:
    plan = QueryPlan.model_validate({
        "intent": "hybrid", "retrieval_query": "cancel policy", "needs_retrieval": True,
        "tool_request": {"name": "get_subscription", "arguments": {"subscription_id": "sub_9"}}})
    ev = make_evidence(1)
    # one transient 503 (retried once, per invariant #7's ONE extra attempt), then sufficient
    grade_script = [ModelUnavailable("503"), EvidenceGrade(sufficient=True, confidence=0.9,
                                                           reason="covers it")]
    snapshot = SubscriptionSnapshot(
        subscription_id="sub_9", customer_id="cus_1", status="active",
        observed_at="2026-09-23T00:00:00Z",
    )
    tool = FakeSubscriptionToolClient([snapshot])

    h = Harness([plan], grade_script, [_answered(ev[0])], ev, tool=tool)
    result = await h.run()

    assert result.status == "answered"
    assert result.tool_executed is True
    assert h.chat_calls <= MAX_GENERATIVE_LLM_CALLS  # plan(1) + grade(retry=2) + answer(1) = 4
    assert h.chat_calls == 4
    assert h.tool_calls <= MAX_TOOL_CALLS
    assert h.tool_calls == 1
    # a transient error keeps its ORIGINAL purpose on retry ("repair" is reserved for a
    # StructuredOutputError re-ask, app/ai/roles.py::_call) -- "grade" appears twice, once per
    # attempt, each with its own model_calls row (invariant #7: every attempt leaves one row).
    assert [r.purpose for r in h.recorder.records] == ["plan", "grade", "grade", "answer"]
    assert [r.status for r in h.recorder.records] == ["ok", "error", "ok", "ok"]
