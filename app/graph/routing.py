"""Pure routing (Plan §8.3): (state) -> next node. Unit-tested without LangGraph.

Any state carrying an `error` goes straight to the fallback. The bounded rewrite lives here:
a rewrite is allowed only while `retrieval_attempts < MAX_RETRIEVAL_ATTEMPTS` and only if the
proposed query is usable.
"""
import re
from dataclasses import dataclass
from typing import Literal

from app.graph.limits import MAX_RETRIEVAL_ATTEMPTS
from app.graph.result import TurnStatus
from app.graph.state import RAGState

_REWRITE_MAX_CHARS = 500


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def is_valid_rewrite(current_query: str, rewritten: str | None) -> bool:
    """Usable = present, non-blank, bounded, and actually different from the current query."""
    if rewritten is None:
        return False
    candidate = _normalise(rewritten)
    if not candidate or len(rewritten.strip()) > _REWRITE_MAX_CHARS:
        return False
    return candidate != _normalise(current_query)


def route_after_understand(state: RAGState) -> Literal["plan", "fallback"]:
    return "fallback" if state.get("error") else "plan"


def route_after_plan(state: RAGState) -> Literal["retrieve", "tool", "fallback"]:
    plan = state.get("plan")
    if state.get("error") or plan is None:
        return "fallback"
    if plan.intent == "clarification":
        return "fallback"
    if plan.needs_retrieval:
        return "retrieve"  # tool (if any) runs after retrieval -- see route_after_retrieve
    if plan.tool_request is not None:
        return "tool"  # pure "subscription" intent: nothing to retrieve, go straight to the tool
    return "fallback"


def route_after_retrieve(state: RAGState) -> Literal["tool", "grade", "fallback"]:
    if state.get("error"):
        return "fallback"
    plan = state.get("plan")
    if plan is not None and plan.tool_request is not None and not state.get("tool_executed"):
        # Tool runs AFTER retrieval on purpose (Plan §18/CLAUDE.md "hybrid" case): build_evidence
        # always keeps the first item of its `new` argument regardless of the cap, so the tool
        # result must be `new` on its OWN build_evidence call, not `existing` on retrieve's -- the
        # tool node is what makes it un-evictable, not the retrieve step.
        return "tool"
    if not state.get("evidence"):
        return "fallback"  # nothing to grade: do not spend a call (Plan §11.2)
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS and \
            state.get("new_evidence_count", 1) == 0:
        return "fallback"  # the rewrite found nothing new: grading identical evidence is waste
    return "grade"


def route_after_tool(state: RAGState) -> Literal["grade", "fallback"]:
    if state.get("error") or not state.get("evidence"):
        return "fallback"  # a 404 with no prior retrieval evidence ends up here: no evidence at all
    return "grade"


def route_after_grade(state: RAGState) -> Literal["generate", "rewrite", "fallback"]:
    grade = state.get("grade")
    if state.get("error") or grade is None:
        return "fallback"
    if grade.sufficient:
        return "generate"
    if state.get("retrieval_attempts", 0) < MAX_RETRIEVAL_ATTEMPTS and is_valid_rewrite(
        state.get("retrieval_query", ""), grade.rewritten_query
    ):
        return "rewrite"
    return "fallback"


def route_after_generate(state: RAGState) -> Literal["validate", "fallback"]:
    # A failed generation has no draft: validating nothing used to raise KeyError (a graph bug that
    # only showed up as a generic "internal error"). It must go to the fallback like any failure.
    if state.get("error") or state.get("draft") is None:
        return "fallback"
    return "validate"


def route_after_validate(state: RAGState) -> Literal["end", "fallback"]:
    draft = state.get("draft")
    if state.get("error") or state.get("citations_valid") is not True:
        return "fallback"  # unknown is never "valid"
    if draft is not None and draft.status != "answered":
        return "fallback"  # clarification / insufficient_evidence drafts are reported, not answered
    return "end"


@dataclass(frozen=True)
class FallbackOutcome:
    status: TurnStatus
    detail: str
    clarification_question: str | None = None
    retry_after_seconds: float | None = None


def fallback_outcome(state: RAGState) -> FallbackOutcome:
    """Why the turn ended without a validated answer. Deterministic; never consults a model.

    A stage that FAILED (plan, embedding/retrieval, grade or generate) is always reported through
    `state["error"]`: `blocked` for invalid output, `temporarily_unavailable` for everything else,
    with the normalised error code as the detail (full table in `app/graph/result.py`). The
    generate stage is no exception — a failed generation leaves no draft, `route_after_generate`
    sends it here, and it ends as e.g. temporarily_unavailable / unavailable, never as an
    unnamed or "internal" outcome.
    """
    error = state.get("error")
    if error is not None:
        if error.kind == "clarification":  # tool 409: ambiguous identifier (Plan §10)
            return FallbackOutcome("clarification", error.code, error.clarification_question)
        status: TurnStatus = "blocked" if error.kind == "blocked" else "temporarily_unavailable"
        return FallbackOutcome(status, error.code, None, error.retry_after_seconds)

    plan = state.get("plan")
    if plan is not None and plan.intent == "clarification":
        return FallbackOutcome("clarification", "planner_clarification",
                               plan.clarification_question)
    if plan is not None and not plan.needs_retrieval and plan.tool_request is None:
        return FallbackOutcome("insufficient_evidence", "no_retrieval_needed")

    draft = state.get("draft")
    if state.get("citations_valid") is False:
        return FallbackOutcome("blocked", "invalid_citations")
    if draft is not None and draft.status == "insufficient_evidence":
        return FallbackOutcome("insufficient_evidence", "generator_insufficient")
    if draft is not None and draft.status == "clarification":
        return FallbackOutcome("clarification", "generator_clarification",
                               draft.clarification_question)

    attempts = state.get("retrieval_attempts", 0)
    if attempts >= MAX_RETRIEVAL_ATTEMPTS and state.get("new_evidence_count", 1) == 0:
        return FallbackOutcome("insufficient_evidence", "rewrite_no_new_evidence")

    grade = state.get("grade")
    if grade is not None and not grade.sufficient:
        proposed = grade.rewritten_query
        if attempts < MAX_RETRIEVAL_ATTEMPTS and proposed is not None and not is_valid_rewrite(
            state.get("retrieval_query", ""), proposed
        ):
            return FallbackOutcome("insufficient_evidence", "rewrite_rejected")
        return FallbackOutcome("insufficient_evidence", "grader_insufficient")

    if not state.get("evidence"):
        return FallbackOutcome("insufficient_evidence", "no_evidence")
    return FallbackOutcome("blocked", "unexpected_state")  # fail closed, never "answered"
