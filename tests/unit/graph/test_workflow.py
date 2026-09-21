"""The Week 4 graph with fake models. Ported from the Week 3 sequential-flow specs.

Ported unchanged: happy path, evidence identity, invented/wrong-version citation, clarification,
tool-only, empty evidence, generator insufficient, tool proposal not executed, 429 (no retry when
retry_after > 10 s), embedding failure, call count. SUPERSEDED by Week 4 policy (called out in the
commit message): weak evidence now rewrites once (was: stop); invalid output now gets one repair
attempt (was: blocked after one call), so those scripts carry two bad outputs.
"""
import asyncio
import random
import uuid
from collections.abc import Sequence

import pytest

from app.ai.chat.fake import FakeChatModelClient, demo_responder
from app.ai.errors import (
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from app.ai.types import ModelMessage, ModelResponse, Usage
from app.retrieval.schemas import Evidence
from tests.unit.ai.helpers import make_evidence
from tests.unit.graph.harness import Harness

POLICY = QueryPlan(intent="policy", retrieval_query="refund window", needs_retrieval=True)
SUFFICIENT = EvidenceGrade(sufficient=True, confidence=0.9, reason="covers it")
WEAK = EvidenceGrade(
    sufficient=False, confidence=0.3, reason="no refund terms", rewritten_query="refund terms")
LEAK = "SECRET-DRAFT-TEXT-MUST-NEVER-SURFACE"
BAD = '{"status": "answered", "answer": "x"'  # truncated JSON


def _ref(e: Evidence) -> CitationRef:
    return CitationRef(document_version_id=e.document_version_id, chunk_id=e.chunk_id)


def _answered(*refs: CitationRef) -> AnswerDraft:
    return AnswerDraft(status="answered", answer=LEAK, citations=list(refs))


# --- happy path & identity (ported) -------------------------------------------------------


async def test_happy_path_makes_exactly_three_calls_and_returns_a_validated_answer() -> None:
    ev = make_evidence(3)
    h = Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[1]))], ev)
    result = await h.run()

    assert result.status == "answered" and result.answer == LEAK
    assert result.citations == (_ref(ev[1]),)
    assert h.chat_calls == 3 and len(h.retrieve_calls) == 1
    assert [r.purpose for r in h.recorder.records] == ["plan", "grade", "answer"]
    assert h.retrieve_calls[0][0] == "refund window"
    (embed_call,) = h.embeddings.calls
    assert embed_call.kind == "query" and embed_call.texts == ["refund window"]
    assert result.retrieval_attempts == 1


async def test_the_answer_model_sees_exactly_the_evidence_that_was_graded() -> None:
    ev = make_evidence(2)
    h = Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[0]))], ev)
    await h.run()
    graded = h.grader.calls[0].messages[-1].content
    answered = h.answerer.calls[0].messages[-1].content
    for e in ev:
        assert str(e.chunk_id) in graded and str(e.chunk_id) in answered


async def test_snapshots_are_built_only_for_cited_and_validated_sources() -> None:
    ev = make_evidence(3)
    result = await Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[1]))], ev).run()
    (snap,) = result.source_snapshots
    assert (snap.document_version_id, snap.chunk_id) == (ev[1].document_version_id, ev[1].chunk_id)
    assert (snap.document_title, snap.version_no, snap.text_snapshot) == (
        ev[1].title, ev[1].version_no, ev[1].text)


# --- citation validation (ported) ---------------------------------------------------------


async def test_citation_outside_the_evidence_blocks_the_draft() -> None:
    ev = make_evidence(2)
    invented = CitationRef(document_version_id=ev[0].document_version_id, chunk_id=uuid.uuid4())
    result = await Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[0]), invented)], ev).run()
    assert result.status == "blocked"
    assert result.answer is None and result.citations == () and result.source_snapshots == ()
    assert LEAK not in repr(result)


async def test_right_chunk_wrong_version_blocks_the_draft() -> None:
    ev = make_evidence(2)
    mixed = CitationRef(document_version_id=ev[1].document_version_id, chunk_id=ev[0].chunk_id)
    result = await Harness([POLICY], [SUFFICIENT], [_answered(mixed)], ev).run()
    assert result.status == "blocked" and LEAK not in repr(result)


# --- routing/branches (ported) ------------------------------------------------------------


async def test_valid_tool_proposal_is_reported_but_never_executed() -> None:
    plan = QueryPlan.model_validate(
        {"intent": "hybrid", "retrieval_query": "cancel policy", "needs_retrieval": True,
         "tool_request": {"name": "get_subscription", "arguments": {"subscription_id": "sub_9"}}})
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
         "tool_request": {"name": "get_subscription", "arguments": {"customer_id": "c1"}}})
    h = Harness([plan])
    result = await h.run()
    assert result.status == "insufficient_evidence"
    assert result.proposed_tool is not None and result.tool_executed is False
    assert h.retrieve_calls == [] and h.chat_calls == 1


async def test_empty_evidence_skips_grader_and_generator() -> None:
    h = Harness([POLICY], evidence=[])
    result = await h.run()
    assert result.status == "insufficient_evidence" and result.detail == "no_evidence"
    assert len(h.grader.calls) == 0 and len(h.answerer.calls) == 0


async def test_generator_may_itself_report_insufficient_evidence() -> None:
    draft = AnswerDraft(status="insufficient_evidence", answer=None)
    result = await Harness([POLICY], [SUFFICIENT], [draft], make_evidence(1)).run()
    assert result.status == "insufficient_evidence" and result.answer is None


# --- failures ----------------------------------------------------------------------------


async def test_provider_rate_limit_with_a_long_retry_after_is_not_retried() -> None:
    limited = ModelRateLimited("429", retry_after_seconds=12.0)
    h = Harness([POLICY], [limited], evidence=make_evidence(1))
    result = await h.run()
    assert result.status == "temporarily_unavailable" and result.retry_after_seconds == 12.0
    assert len(h.grader.calls) == 1 and len(h.answerer.calls) == 0
    assert h.recorder.records[-1].error_code == "rate_limited"


async def test_embedding_failure_becomes_temporarily_unavailable() -> None:
    from app.ai.embeddings.fake import FakeEmbeddingClient

    h = Harness([POLICY], evidence=make_evidence(1))
    h.embeddings = FakeEmbeddingClient(fail_with=ModelRateLimited("429", retry_after_seconds=None))
    result = await h.run()
    assert result.status == "temporarily_unavailable" and h.retrieve_calls == []
    assert len(h.embeddings.calls) == 2  # one transient retry, outside the generative pool
    assert h.budget.generative_calls == 1  # only the planner consumed the pool


async def test_query_embedding_recovers_after_one_transient_failure() -> None:
    from app.ai.embeddings.fake import FakeEmbeddingClient

    ev = make_evidence(1)
    h = Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[0]))], ev)
    h.embeddings = FakeEmbeddingClient(fail_with=[ModelUnavailable("503")])
    assert (await h.run()).status == "answered"


# --- repair / retry (superseded specs + new) ---------------------------------------------


async def test_invalid_planner_output_is_repaired_once_then_blocked_if_still_invalid() -> None:
    bad_tool = {"intent": "hybrid", "retrieval_query": "q", "needs_retrieval": True,
                "tool_request": {"name": "cancel_subscription",
                                 "arguments": {"subscription_id": "s"}}}
    h = Harness([bad_tool, bad_tool], evidence=make_evidence(1))
    result = await h.run()
    assert result.status == "blocked"
    assert h.retrieve_calls == [] and h.embeddings.calls == []
    assert h.chat_calls == 2  # plan + repair
    assert [r.purpose for r in h.recorder.records] == ["plan", "repair"]


async def test_planner_output_that_repairs_successfully_proceeds() -> None:
    ev = make_evidence(1)
    h = Harness([BAD, POLICY], [SUFFICIENT], [_answered(_ref(ev[0]))], ev)
    result = await h.run()
    assert result.status == "answered" and h.chat_calls == 4
    assert [r.purpose for r in h.recorder.records] == ["plan", "repair", "grade", "answer"]


async def test_generator_invalid_json_is_repaired_once_then_blocked() -> None:
    h = Harness([POLICY], [SUFFICIENT], [BAD, BAD], make_evidence(1))
    result = await h.run()
    assert result.status == "blocked" and result.answer is None
    assert h.recorder.records[-1].error_code == "structured_output_invalid"
    assert [r.purpose for r in h.recorder.records][-2:] == ["answer", "repair"]


async def test_a_503_on_the_answer_step_is_retried_once() -> None:
    ev = make_evidence(1)
    h = Harness([POLICY], [SUFFICIENT], [ModelUnavailable("503"), _answered(_ref(ev[0]))], ev)
    result = await h.run()
    assert result.status == "answered" and h.clock.sleeps == [2.0]
    assert h.budget.generative_calls == 4


async def test_two_503s_on_the_answer_step_end_as_temporarily_unavailable() -> None:
    h = Harness([POLICY], [SUFFICIENT], [ModelUnavailable("503"), ModelUnavailable("503")],
                make_evidence(1))
    result = await h.run()
    assert result.status == "temporarily_unavailable" and result.answer is None


# --- bounded rewrite ----------------------------------------------------------------------


async def test_weak_evidence_rewrites_once_and_then_answers_from_the_union() -> None:
    first, second = make_evidence(2), make_evidence(2, text_prefix="better")
    h = Harness([POLICY], [WEAK, SUFFICIENT], [_answered(_ref(second[0]))],
                retrievals=[first, second])
    result = await h.run()

    assert result.status == "answered" and result.retrieval_attempts == 2
    assert [q for q, _, _ in h.retrieve_calls] == ["refund window", "refund terms"]
    assert [r.purpose for r in h.recorder.records] == ["plan", "grade", "grade", "answer"]
    assert h.chat_calls == 4
    assert len(result.evidence) == 4  # union of both attempts
    assert h.phases == ["planning", "retrieving", "grading", "retrieving", "grading", "generating"]


async def test_rewrite_is_embedded_as_a_query_too() -> None:
    first, second = make_evidence(1), make_evidence(1)
    h = Harness([POLICY], [WEAK, SUFFICIENT], [_answered(_ref(second[0]))],
                retrievals=[first, second])
    await h.run()
    assert [c.texts for c in h.embeddings.calls] == [["refund window"], ["refund terms"]]
    assert {c.kind for c in h.embeddings.calls} == {"query"}


async def test_a_grader_that_always_says_weak_cannot_loop() -> None:
    always_weak = [
        EvidenceGrade(sufficient=False, confidence=0.1, reason="r", rewritten_query="second"),
        EvidenceGrade(sufficient=False, confidence=0.1, reason="r", rewritten_query="third"),
    ]
    h = Harness([POLICY], always_weak, retrievals=[make_evidence(1), make_evidence(1)])
    result = await h.run()
    assert result.status == "insufficient_evidence" and result.detail == "grader_insufficient"
    assert len(h.retrieve_calls) == 2 and h.chat_calls == 3  # plan + grade + grade
    assert len(h.answerer.calls) == 0


async def test_rewrite_identical_to_the_original_query_is_rejected() -> None:
    same = EvidenceGrade(sufficient=False, confidence=0.1, reason="r",
                         rewritten_query="  REFUND   window ")
    h = Harness([POLICY], [same], evidence=make_evidence(1))
    result = await h.run()
    assert result.status == "insufficient_evidence" and result.detail == "rewrite_rejected"
    assert len(h.retrieve_calls) == 1


async def test_weak_evidence_without_a_rewrite_stops() -> None:
    h = Harness([POLICY], [EvidenceGrade(sufficient=False, confidence=0.1, reason="r")],
                evidence=make_evidence(2))
    result = await h.run()
    assert result.status == "insufficient_evidence" and result.detail == "grader_insufficient"
    assert len(h.retrieve_calls) == 1 and result.grade is not None


async def test_rewrite_that_finds_nothing_new_does_not_spend_a_second_grade() -> None:
    ev = make_evidence(2)
    h = Harness([POLICY], [WEAK], retrievals=[ev, ev])
    result = await h.run()
    assert result.status == "insufficient_evidence" and result.detail == "rewrite_no_new_evidence"
    assert h.chat_calls == 2 and len(h.retrieve_calls) == 2


# --- pool bound under adversarial failure (invariant #7) ---------------------------------


async def test_when_repairs_use_up_the_pool_the_last_invalid_output_is_final() -> None:
    # plan(bad)+repair(ok) + grade(weak) + grade2(bad) = 4 calls; no unit left to repair grade2
    ev1, ev2 = make_evidence(1), make_evidence(1)
    h = Harness([BAD, POLICY], [WEAK, BAD, BAD], [_answered(_ref(ev2[0]))],
                retrievals=[ev1, ev2])
    result = await h.run()
    assert h.chat_calls == 4 and h.budget.generative_calls == 4
    assert result.status == "blocked" and result.detail == "structured_output_invalid"
    assert len(h.answerer.calls) == 0


async def test_a_call_that_cannot_get_a_unit_ends_the_turn_before_any_request() -> None:
    # plan(bad)+repair(ok) + grade(weak) + grade2(ok) = 4 calls; answer finds the pool empty
    ev1, ev2 = make_evidence(1), make_evidence(1)
    h = Harness([BAD, POLICY], [WEAK, SUFFICIENT], [_answered(_ref(ev2[0]))],
                retrievals=[ev1, ev2])
    result = await h.run()
    assert result.status == "temporarily_unavailable" and result.detail == "budget_exhausted"
    assert h.chat_calls == 4 and len(h.answerer.calls) == 0  # no provider request was made
    assert result.answer is None


def _fuzz_harness(seed: int) -> tuple[Harness, dict[str, int]]:
    rng = random.Random(seed)
    counter = {"n": 0}

    def responder(messages: Sequence[ModelMessage], model: type, purpose: str):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        roll = rng.random()
        if roll < 0.15:
            return BAD
        if roll < 0.28:
            retry_after = rng.choice([None, 1.0, 20.0])
            return rng.choice([ModelUnavailable("503"), ModelTimeout("t"),
                               ModelRateLimited("429", retry_after_seconds=retry_after)])
        if model is EvidenceGrade and roll < 0.6:
            return EvidenceGrade(sufficient=False, confidence=0.2, reason="r",
                                 rewritten_query=f"rewrite {rng.random()}")
        if model is AnswerDraft and roll < 0.42:
            return AnswerDraft(status="answered", answer=LEAK,
                               citations=[CitationRef(document_version_id=uuid.uuid4(),
                                                      chunk_id=uuid.uuid4())])
        return demo_responder(messages, model, purpose)  # type: ignore[arg-type]

    def fake() -> FakeChatModelClient:
        return FakeChatModelClient(responder=responder)

    evidence = [make_evidence(rng.randint(0, 3)) for _ in range(2)]
    h = Harness(planner=fake(), grader=fake(), answerer=fake(), retrievals=evidence)
    return h, counter


@pytest.mark.parametrize("seed", range(300))
async def test_no_failure_sequence_can_exceed_the_hard_limits(seed: int) -> None:
    h, counter = _fuzz_harness(seed)
    result = await h.run()

    assert counter["n"] <= 4, "MAX_GENERATIVE_LLM_CALLS exceeded"
    assert h.budget.generative_calls == counter["n"]  # every provider call was acquired first
    assert len(h.retrieve_calls) <= 2, "MAX_RETRIEVAL_ATTEMPTS exceeded"
    assert result.status in {"answered", "clarification", "insufficient_evidence",
                             "temporarily_unavailable", "blocked"}
    # a graph bug must not hide behind a legitimate-looking fallback status
    assert result.detail not in {"internal_error", "unexpected_state", "graph_recursion_limit"}
    if result.status != "answered":
        assert result.answer is None and result.citations == () and LEAK not in repr(result)
    else:
        allowed = {(e.document_version_id, e.chunk_id) for e in result.evidence}
        assert {(c.document_version_id, c.chunk_id) for c in result.citations} <= allowed


# --- graph timeout & phases ---------------------------------------------------------------


class _SlowChat:
    provider = "slow"
    model_name = "slow"

    async def complete_structured(self, *, messages, response_model, purpose, timeout_seconds):  # type: ignore[no-untyped-def]
        await asyncio.sleep(5)
        return ModelResponse(POLICY, "slow", "slow", Usage(1, 1), 5000)


async def test_graph_timeout_ends_the_turn_as_temporarily_unavailable() -> None:
    h = Harness(planner=_SlowChat())
    result = await h.run(timeout_seconds=0.05)
    assert result.status == "temporarily_unavailable" and result.detail == "graph_timeout"
    assert result.answer is None


async def test_only_phase_events_are_emitted_and_none_carry_answer_text() -> None:
    ev = make_evidence(1)
    h = Harness([POLICY], [SUFFICIENT], [_answered(_ref(ev[0]))], ev)
    await h.run()
    assert h.phases == ["planning", "retrieving", "grading", "generating"]
    assert all(p in {"planning", "retrieving", "grading", "generating"} for p in h.phases)
    assert LEAK not in repr(h.phases)


@pytest.mark.parametrize("question", ["", "   \n\t ", "x" * 2001])
async def test_empty_or_oversized_questions_never_reach_a_model(question: str) -> None:
    h = Harness([POLICY], evidence=make_evidence(1))
    result = await h.run(question)
    assert result.status == "blocked" and result.detail == "invalid_question"
    assert h.chat_calls == 0 and h.embeddings.calls == [] and h.retrieve_calls == []


async def test_whitespace_in_the_question_is_normalised_before_planning() -> None:
    h = Harness([POLICY], evidence=[])
    await h.run("  How   long\n is the   refund window? ")
    assert "Question:\nHow long is the refund window?" in h.planner.calls[0].messages[-1].content
