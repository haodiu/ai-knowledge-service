"""Routing is pure: (state) -> next node. Every edge of Plan §8.3, plus the rewrite guards."""
import pytest

from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from app.graph.routing import (
    fallback_outcome,
    is_valid_rewrite,
    route_after_generate,
    route_after_grade,
    route_after_plan,
    route_after_retrieve,
    route_after_understand,
    route_after_validate,
)
from app.graph.state import TurnError
from tests.unit.ai.helpers import make_evidence

POLICY = QueryPlan(intent="policy", retrieval_query="refund window", needs_retrieval=True)
CLARIFY = QueryPlan(intent="clarification", retrieval_query="x", needs_retrieval=False,
                    clarification_question="Which plan?")
TOOL_ONLY = QueryPlan.model_validate(
    {"intent": "subscription", "retrieval_query": "my sub", "needs_retrieval": False,
     "tool_request": {"name": "get_subscription", "arguments": {"customer_id": "c"}}})


def _grade(sufficient: bool, rewrite: str | None = None) -> EvidenceGrade:
    return EvidenceGrade(sufficient=sufficient, confidence=0.5, reason="r", rewritten_query=rewrite)


# --- after plan ---------------------------------------------------------------------------


def test_plan_routes() -> None:
    assert route_after_plan({"plan": POLICY}) == "retrieve"
    assert route_after_plan({"plan": CLARIFY}) == "fallback"
    assert route_after_plan({"plan": TOOL_ONLY}) == "fallback"  # nothing to retrieve, tool is W7


def test_any_error_goes_to_fallback_from_every_router() -> None:
    err = TurnError(kind="unavailable", code="unavailable")
    assert route_after_plan({"error": err, "plan": POLICY}) == "fallback"
    assert route_after_retrieve({"error": err, "evidence": make_evidence(1)}) == "fallback"
    assert route_after_grade({"error": err, "grade": _grade(True)}) == "fallback"
    assert route_after_validate({"error": err, "citations_valid": True}) == "fallback"


# --- after retrieve -----------------------------------------------------------------------


def test_empty_evidence_never_reaches_the_grader() -> None:
    assert route_after_retrieve({"evidence": [], "retrieval_attempts": 1}) == "fallback"


def test_evidence_goes_to_the_grader() -> None:
    assert route_after_retrieve(
        {"evidence": make_evidence(2), "retrieval_attempts": 1, "new_evidence_count": 2}) == "grade"


def test_a_rewrite_that_finds_nothing_new_skips_the_second_grade() -> None:
    state = {"evidence": make_evidence(2), "retrieval_attempts": 2, "new_evidence_count": 0}
    assert route_after_retrieve(state) == "fallback"
    assert fallback_outcome(state).detail == "rewrite_no_new_evidence"  # type: ignore[arg-type]


def test_a_rewrite_that_finds_something_new_is_graded_again() -> None:
    state = {"evidence": make_evidence(3), "retrieval_attempts": 2, "new_evidence_count": 1}
    assert route_after_retrieve(state) == "grade"  # type: ignore[arg-type]


# --- after grade --------------------------------------------------------------------------


def test_sufficient_evidence_goes_to_generation() -> None:
    assert route_after_grade({"grade": _grade(True), "retrieval_attempts": 1,
                              "retrieval_query": "q"}) == "generate"


def test_weak_first_attempt_with_a_usable_rewrite_loops_once() -> None:
    state = {"grade": _grade(False, "refund terms"), "retrieval_attempts": 1,
             "retrieval_query": "refund window"}
    assert route_after_grade(state) == "rewrite"  # type: ignore[arg-type]


def test_weak_second_attempt_never_rewrites_again() -> None:
    state = {"grade": _grade(False, "another query"), "retrieval_attempts": 2,
             "retrieval_query": "refund terms"}
    assert route_after_grade(state) == "fallback"  # type: ignore[arg-type]
    assert fallback_outcome(state).detail == "grader_insufficient"  # type: ignore[arg-type]


@pytest.mark.parametrize("rewrite", [None, "", "   ", "REFUND   Window", "refund window"])
def test_unusable_rewrites_are_rejected(rewrite: str | None) -> None:
    grade = EvidenceGrade.model_construct(  # bypass schema so blank/oversized reach the router
        sufficient=False, confidence=0.5, reason="r", missing_information=[],
        rewritten_query=rewrite)
    state = {"grade": grade, "retrieval_attempts": 1, "retrieval_query": "refund window"}
    assert route_after_grade(state) == "fallback"  # type: ignore[arg-type]


def test_overlong_rewrite_is_rejected() -> None:
    grade = EvidenceGrade.model_construct(
        sufficient=False, confidence=0.5, reason="r", missing_information=[],
        rewritten_query="x" * 501)
    state = {"grade": grade, "retrieval_attempts": 1, "retrieval_query": "q"}
    assert route_after_grade(state) == "fallback"  # type: ignore[arg-type]
    assert fallback_outcome(state).detail == "rewrite_rejected"  # type: ignore[arg-type]


def test_is_valid_rewrite_is_whitespace_and_case_insensitive() -> None:
    assert not is_valid_rewrite("Refund Window", "  refund   window ")
    assert is_valid_rewrite("refund window", "refund terms")
    assert not is_valid_rewrite("q", None)


# --- after validate -----------------------------------------------------------------------


def test_validation_routes() -> None:
    assert route_after_validate({"citations_valid": True}) == "end"
    assert route_after_validate({"citations_valid": False}) == "fallback"
    assert route_after_validate({}) == "fallback"  # unknown is never "valid"


# --- fallback outcomes --------------------------------------------------------------------


def test_fallback_outcomes() -> None:
    ev = make_evidence(1)
    assert fallback_outcome({"plan": CLARIFY}).status == "clarification"  # type: ignore[arg-type]
    assert fallback_outcome({"plan": CLARIFY}).clarification_question == "Which plan?"  # type: ignore[arg-type]
    assert fallback_outcome({"plan": TOOL_ONLY}).detail == "no_retrieval_needed"  # type: ignore[arg-type]
    assert fallback_outcome({"plan": POLICY, "evidence": []}).detail == "no_evidence"  # type: ignore[arg-type]
    blocked = fallback_outcome({"plan": POLICY, "evidence": ev, "grade": _grade(True),  # type: ignore[arg-type]
                                "citations_valid": False})
    assert (blocked.status, blocked.detail) == ("blocked", "invalid_citations")
    gen_insufficient = fallback_outcome(  # type: ignore[arg-type]
        {"plan": POLICY, "evidence": ev, "grade": _grade(True), "citations_valid": True,
         "draft": AnswerDraft(status="insufficient_evidence")})
    assert gen_insufficient.detail == "generator_insufficient"
    unavailable = fallback_outcome(  # type: ignore[arg-type]
        {"error": TurnError(kind="unavailable", code="rate_limited", retry_after_seconds=4.0)})
    assert (unavailable.status, unavailable.retry_after_seconds) == ("temporarily_unavailable", 4.0)
    invalid = fallback_outcome(  # type: ignore[arg-type]
        {"error": TurnError(kind="blocked", code="structured_output_invalid")})
    assert invalid.status == "blocked"
    _ = CitationRef  # imported for parity with the workflow tests


# --- understand / generate ----------------------------------------------------------------


def test_understand_and_generate_routes() -> None:
    err = TurnError(kind="blocked", code="invalid_question")
    assert route_after_understand({}) == "plan"
    assert route_after_understand({"error": err}) == "fallback"
    draft = AnswerDraft(status="insufficient_evidence")
    assert route_after_generate({"draft": draft}) == "validate"
    assert route_after_generate({}) == "fallback"  # a failed generation has no draft
    assert route_after_generate({"error": TurnError(kind="unavailable", code="x"),
                                 "draft": draft}) == "fallback"


async def test_retrieve_node_refuses_a_third_attempt_even_if_routing_were_wrong() -> None:
    """Defence in depth: the node re-checks MAX_RETRIEVAL_ATTEMPTS behind route_after_grade."""
    from app.graph.nodes.retrieval import retrieve_node
    from tests.unit.graph.harness import Harness

    h = Harness(evidence=make_evidence(1))
    out = await retrieve_node(  # type: ignore[arg-type]
        {"retrieval_query": "q", "retrieval_attempts": 2, "evidence": make_evidence(1)}, h.ctx())
    assert out["error"].code == "retrieval_attempts_exceeded"
    assert h.retrieve_calls == [] and h.embeddings.calls == []  # nothing was spent
