"""The Week 3 sequential flow (plan -> retrieve -> grade -> generate -> validate), fake clients.

Every branch that must NOT reach the user (invented citation, invalid output, provider failure)
has a test asserting that no draft text leaks into the result.
"""
import uuid
from collections.abc import Sequence

import pytest

from app.ai.chat.fake import FakeChatModelClient
from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.errors import ModelRateLimited
from app.ai.pipeline import TurnResult, run_turn
from app.ai.prompts.loader import load_prompts
from app.ai.recorder import InMemoryModelCallRecorder
from app.ai.registry import ModelRegistry
from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from app.retrieval.schemas import Evidence
from tests.unit.ai.helpers import make_evidence

POLICY = QueryPlan(intent="policy", retrieval_query="refund window", needs_retrieval=True)
SUFFICIENT = EvidenceGrade(sufficient=True, confidence=0.9, reason="covers it")
WEAK = EvidenceGrade(
    sufficient=False, confidence=0.3, reason="no refund terms", rewritten_query="refund terms"
)
LEAK = "SECRET-DRAFT-TEXT-MUST-NEVER-SURFACE"


def _ref(e: Evidence) -> CitationRef:
    return CitationRef(document_version_id=e.document_version_id, chunk_id=e.chunk_id)


def _answered(*refs: CitationRef) -> AnswerDraft:
    return AnswerDraft(status="answered", answer=LEAK, citations=list(refs))


class Harness:
    def __init__(self, plan=(), grade=(), answer=(), evidence: Sequence[Evidence] = ()) -> None:  # type: ignore[no-untyped-def]
        self.planner = FakeChatModelClient(list(plan))
        self.grader = FakeChatModelClient(list(grade))
        self.answerer = FakeChatModelClient(list(answer))
        self.embeddings = FakeEmbeddingClient()
        self.recorder = InMemoryModelCallRecorder()
        self.evidence = list(evidence)
        self.retrieve_calls: list[tuple[str, int]] = []

    async def _retrieve(self, query: str, embedding: Sequence[float]) -> Sequence[Evidence]:
        self.retrieve_calls.append((query, len(embedding)))
        return self.evidence

    async def run(self, question: str = "How long do I have to ask for a refund?") -> TurnResult:
        return await run_turn(
            question=question,
            models=ModelRegistry(self.planner, self.grader, self.answerer, self.embeddings),
            prompts=load_prompts("v1"),
            recorder=self.recorder,
            retrieve=self._retrieve,
            embedding_model="fake-embed",
            timeout_seconds=5,
        )

    @property
    def chat_calls(self) -> int:
        return len(self.planner.calls) + len(self.grader.calls) + len(self.answerer.calls)


async def test_happy_path_makes_exactly_three_calls_and_returns_a_validated_answer() -> None:
    ev = make_evidence(3)
    h = Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[1]))], ev)
    result = await h.run()

    assert result.status == "answered" and result.answer == LEAK
    assert result.citations == (_ref(ev[1]),)
    assert h.chat_calls == 3 and len(h.retrieve_calls) == 1
    assert [r.purpose for r in h.recorder.records] == ["plan", "grade", "answer"]
    assert h.retrieve_calls[0][0] == "refund window"  # the planner's retrieval_query
    (embed_call,) = h.embeddings.calls
    assert embed_call.kind == "query" and embed_call.texts == ["refund window"]


async def test_the_answer_model_sees_exactly_the_evidence_that_was_graded() -> None:
    ev = make_evidence(2)
    h = Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[0]))], ev)
    await h.run()
    graded = h.grader.calls[0].messages[-1].content
    answered = h.answerer.calls[0].messages[-1].content
    for e in ev:
        assert str(e.chunk_id) in graded and str(e.chunk_id) in answered


async def test_citation_outside_the_evidence_blocks_the_draft() -> None:
    ev = make_evidence(2)
    invented = CitationRef(document_version_id=ev[0].document_version_id, chunk_id=uuid.uuid4())
    h = Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[0]), invented)], ev)
    result = await h.run()

    assert result.status == "blocked"
    assert result.answer is None and result.citations == ()
    assert LEAK not in repr(result)  # the unvalidated draft is not reachable from the result


async def test_right_chunk_wrong_version_blocks_the_draft() -> None:
    ev = make_evidence(2)
    mixed = CitationRef(document_version_id=ev[1].document_version_id, chunk_id=ev[0].chunk_id)
    result = await Harness([POLICY], [SUFFICIENT], [_answered(mixed)], ev).run()
    assert result.status == "blocked" and LEAK not in repr(result)


async def test_generator_returning_invalid_json_is_blocked_not_used() -> None:
    ev = make_evidence(1)
    h = Harness([POLICY], [SUFFICIENT], ['{"status": "answered", "answer": "x"'], ev)
    result = await h.run()
    assert result.status == "blocked" and result.answer is None
    assert h.recorder.records[-1].error_code == "structured_output_invalid"


async def test_planner_proposing_an_unknown_tool_is_rejected_before_any_retrieval() -> None:
    bad = {"intent": "hybrid", "retrieval_query": "q", "needs_retrieval": True,
           "tool_request": {"name": "cancel_subscription", "arguments": {"subscription_id": "s"}}}
    h = Harness([bad], evidence=make_evidence(1))
    result = await h.run()
    assert result.status == "blocked"
    assert h.retrieve_calls == [] and h.embeddings.calls == []
    assert h.chat_calls == 1


async def test_valid_tool_proposal_is_reported_but_never_executed() -> None:
    plan = QueryPlan.model_validate(
        {"intent": "hybrid", "retrieval_query": "cancel policy", "needs_retrieval": True,
         "tool_request": {"name": "get_subscription", "arguments": {"subscription_id": "sub_9"}}}
    )
    ev = make_evidence(1)
    result = await Harness([plan], [SUFFICIENT], [_answered(_ref(ev[0]))], ev).run()
    assert result.status == "answered"
    assert result.proposed_tool is not None and result.proposed_tool.name == "get_subscription"
    assert result.tool_executed is False


async def test_clarification_plan_stops_before_retrieval_and_grading() -> None:
    plan = QueryPlan(intent="clarification", retrieval_query="unclear", needs_retrieval=False,
                     clarification_question="Which plan do you mean?")
    h = Harness([plan], evidence=make_evidence(1))
    result = await h.run()
    assert result.status == "clarification"
    assert result.clarification_question == "Which plan do you mean?"
    assert h.retrieve_calls == [] and h.chat_calls == 1


async def test_tool_only_plan_without_retrieval_is_insufficient_evidence() -> None:
    plan = QueryPlan.model_validate(
        {"intent": "subscription", "retrieval_query": "my sub", "needs_retrieval": False,
         "tool_request": {"name": "get_subscription", "arguments": {"customer_id": "c1"}}}
    )
    h = Harness([plan])
    result = await h.run()
    assert result.status == "insufficient_evidence"
    assert result.proposed_tool is not None and result.tool_executed is False
    assert h.retrieve_calls == [] and h.chat_calls == 1


async def test_empty_evidence_skips_grader_and_generator() -> None:
    h = Harness([POLICY], evidence=[])
    result = await h.run()
    assert result.status == "insufficient_evidence"
    assert len(h.grader.calls) == 0 and len(h.answerer.calls) == 0


async def test_weak_evidence_stops_without_rewrite_or_generation() -> None:
    h = Harness([POLICY], [WEAK], evidence=make_evidence(2))
    result = await h.run()
    assert result.status == "insufficient_evidence"
    assert len(h.answerer.calls) == 0
    assert len(h.retrieve_calls) == 1  # Week 3 has NO rewrite loop; rewritten_query is not used
    assert result.grade == WEAK


async def test_generator_may_itself_report_insufficient_evidence() -> None:
    draft = AnswerDraft(status="insufficient_evidence", answer=None)
    result = await Harness([POLICY], [SUFFICIENT], [draft], make_evidence(1)).run()
    assert result.status == "insufficient_evidence" and result.answer is None


async def test_provider_rate_limit_becomes_temporarily_unavailable_without_retry() -> None:
    limited = ModelRateLimited("429", retry_after_seconds=12.0)
    h = Harness([POLICY], [limited], evidence=make_evidence(1))
    result = await h.run()
    assert result.status == "temporarily_unavailable"
    assert result.retry_after_seconds == 12.0
    assert len(h.grader.calls) == 1 and len(h.answerer.calls) == 0  # no hidden retry
    assert h.recorder.records[-1].error_code == "rate_limited"


async def test_embedding_failure_becomes_temporarily_unavailable() -> None:
    h = Harness([POLICY], evidence=make_evidence(1))
    h.embeddings = FakeEmbeddingClient(fail_with=ModelRateLimited("429", retry_after_seconds=None))
    result = await h.run()
    assert result.status == "temporarily_unavailable"
    assert h.retrieve_calls == []


@pytest.mark.parametrize("n_evidence", [1, 8])
async def test_a_turn_never_exceeds_three_chat_calls_in_week_3(n_evidence: int) -> None:
    ev = make_evidence(n_evidence)
    h = Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[0]))], ev)
    await h.run()
    assert h.chat_calls == 3
