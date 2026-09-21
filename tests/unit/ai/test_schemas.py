"""QueryPlan / EvidenceGrade / AnswerDraft must reject malformed model output (Plan §11.2, §16.1).

These are the release-blocking negative tests for "invalid output is never used directly".
"""
import json
import uuid

import pytest
from pydantic import ValidationError

from app.ai.schemas import (
    TOOL_ALLOWLIST,
    AnswerDraft,
    CitationRef,
    EvidenceGrade,
    QueryPlan,
    ToolRequest,
)


def _plan(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "intent": "policy",
        "retrieval_query": "refund window",
        "needs_retrieval": True,
    }
    base.update(over)
    return base


def _tool(**args: object) -> dict[str, object]:
    return {"name": "get_subscription", "arguments": args}


# --- ToolRequest / allowlist ---------------------------------------------------------------


def test_allowlist_is_exactly_the_read_only_subscription_tool() -> None:
    assert TOOL_ALLOWLIST == frozenset({"get_subscription"})


@pytest.mark.parametrize("name", ["delete_subscription", "cancel_subscription", "get_user", ""])
def test_tool_outside_allowlist_is_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        ToolRequest.model_validate({"name": name, "arguments": {"subscription_id": "sub_1"}})


def test_tool_arguments_need_exactly_one_identifier() -> None:
    ToolRequest.model_validate(_tool(subscription_id="sub_1"))
    ToolRequest.model_validate(_tool(customer_id="cus_1"))
    for bad in ({}, {"subscription_id": "a", "customer_id": "b"}):
        with pytest.raises(ValidationError):
            ToolRequest.model_validate(_tool(**bad))


def test_tool_arguments_reject_unknown_fields_and_overlong_ids() -> None:
    with pytest.raises(ValidationError):
        ToolRequest.model_validate(_tool(subscription_id="s", url="http://evil"))  # no LLM URLs
    with pytest.raises(ValidationError):
        ToolRequest.model_validate(_tool(subscription_id="x" * 65))
    with pytest.raises(ValidationError):
        ToolRequest.model_validate(_tool(subscription_id="   "))


# --- QueryPlan -----------------------------------------------------------------------------


def test_valid_policy_plan_parses() -> None:
    plan = QueryPlan.model_validate(_plan())
    assert plan.tool_request is None and plan.needs_retrieval


def test_retrieval_query_is_bounded_and_not_blank() -> None:
    QueryPlan.model_validate(_plan(retrieval_query="x" * 500))
    for bad in ("x" * 501, "", "   "):
        with pytest.raises(ValidationError):
            QueryPlan.model_validate(_plan(retrieval_query=bad))


def test_plan_rejects_unknown_intent_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(_plan(intent="write_data"))
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(_plan(sql="SELECT 1"))  # the LLM never supplies SQL


def test_plan_with_unknown_tool_is_rejected_as_a_whole() -> None:
    bad_tool = {"name": "drop_table", "arguments": {"subscription_id": "s"}}
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(_plan(intent="hybrid", tool_request=bad_tool))


def test_plan_intent_must_agree_with_tool_and_retrieval_flags() -> None:
    tool = _tool(subscription_id="sub_1")
    # policy must not carry a tool
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(_plan(intent="policy", tool_request=tool))
    # subscription / hybrid must carry one
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(_plan(intent="subscription", needs_retrieval=False))
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(_plan(intent="hybrid"))
    # hybrid must retrieve
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(_plan(intent="hybrid", needs_retrieval=False, tool_request=tool))
    # policy must retrieve
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(_plan(needs_retrieval=False))

    QueryPlan.model_validate(_plan(intent="subscription", needs_retrieval=False, tool_request=tool))
    QueryPlan.model_validate(_plan(intent="hybrid", tool_request=tool))


def test_clarification_plan_needs_a_question_and_nothing_else() -> None:
    ok = _plan(intent="clarification", needs_retrieval=False, clarification_question="Which plan?")
    QueryPlan.model_validate(ok)
    for bad in (
        {**ok, "clarification_question": None},
        {**ok, "clarification_question": "  "},
        {**ok, "needs_retrieval": True},
        {**ok, "tool_request": _tool(subscription_id="s")},
    ):
        with pytest.raises(ValidationError):
            QueryPlan.model_validate(bad)


# --- EvidenceGrade -------------------------------------------------------------------------


def _grade(**over: object) -> dict[str, object]:
    base: dict[str, object] = {"sufficient": True, "confidence": 0.8, "reason": "covers refunds"}
    base.update(over)
    return base


def test_valid_grade_parses_and_defaults() -> None:
    grade = EvidenceGrade.model_validate(_grade())
    assert grade.missing_information == [] and grade.rewritten_query is None


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 5, float("nan"), float("inf")])
def test_confidence_outside_unit_range_is_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError):
        EvidenceGrade.model_validate(_grade(confidence=confidence))


def test_confidence_nan_in_raw_json_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidenceGrade.model_validate_json('{"sufficient": true, "confidence": NaN, "reason": "r"}')


def test_grade_bounds_on_lengths() -> None:
    EvidenceGrade.model_validate(_grade(reason="r" * 500, rewritten_query="q" * 500))
    for bad in (
        _grade(reason="r" * 501),
        _grade(rewritten_query="q" * 501),
        _grade(rewritten_query="   "),
        _grade(missing_information=["m"] * 6),
        _grade(missing_information=["m" * 201]),
        _grade(unexpected=True),
    ):
        with pytest.raises(ValidationError):
            EvidenceGrade.model_validate(bad)


# --- AnswerDraft ---------------------------------------------------------------------------


def _cite() -> dict[str, str]:
    return {"document_version_id": str(uuid.uuid4()), "chunk_id": str(uuid.uuid4())}


def test_answered_draft_parses() -> None:
    draft = AnswerDraft.model_validate(
        {"status": "answered", "answer": "14 days.", "citations": [_cite()]}
    )
    assert isinstance(draft.citations[0], CitationRef)


def test_answered_draft_needs_text_and_at_least_one_citation() -> None:
    for bad in (
        {"status": "answered", "answer": "x", "citations": []},
        {"status": "answered", "answer": None, "citations": [_cite()]},
        {"status": "answered", "answer": "  ", "citations": [_cite()]},
    ):
        with pytest.raises(ValidationError):
            AnswerDraft.model_validate(bad)


def test_non_answer_statuses_carry_no_citations() -> None:
    AnswerDraft.model_validate({"status": "insufficient_evidence", "answer": None})
    AnswerDraft.model_validate(
        {"status": "clarification", "answer": None, "clarification_question": "Which plan?"}
    )
    for bad in (
        {"status": "insufficient_evidence", "answer": None, "citations": [_cite()]},
        {"status": "clarification", "answer": None, "citations": [_cite()],
         "clarification_question": "Which?"},
        {"status": "clarification", "answer": None},  # no question
    ):
        with pytest.raises(ValidationError):
            AnswerDraft.model_validate(bad)


def test_citations_must_be_uuids_and_bounded() -> None:
    with pytest.raises(ValidationError):
        AnswerDraft.model_validate(
            {"status": "answered", "answer": "x",
             "citations": [{"document_version_id": "not-a-uuid", "chunk_id": str(uuid.uuid4())}]}
        )
    with pytest.raises(ValidationError):
        AnswerDraft.model_validate(
            {"status": "answered", "answer": "x", "citations": [_cite() for _ in range(9)]}
        )
    with pytest.raises(ValidationError):
        AnswerDraft.model_validate({"status": "answered", "answer": "x", "citations": [_cite()],
                                    "extra": 1})


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AnswerDraft.model_validate({"status": "maybe", "answer": "x"})


def test_schemas_roundtrip_through_json() -> None:
    draft = AnswerDraft.model_validate(
        {"status": "answered", "answer": "x", "citations": [_cite()]}
    )
    assert AnswerDraft.model_validate_json(draft.model_dump_json()) == draft
    assert json.loads(draft.model_dump_json())["status"] == "answered"
