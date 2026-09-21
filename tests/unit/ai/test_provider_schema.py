"""to_provider_schema: Pydantic JSON Schema -> the subset OpenAI-compatible providers accept.

The provider schema is only a hint to the model; parse_structured/Pydantic stays the real gate.
"""
from typing import Any

import pytest

from app.ai.chat.schema import to_provider_schema
from app.ai.schemas import AnswerDraft, EvidenceGrade, QueryPlan

BANNED = {"$ref", "$defs", "title", "default", "additionalProperties", "format", "anyOf", "allOf"}


def _walk(node: Any):  # type: ignore[no-untyped-def]
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("model", [QueryPlan, EvidenceGrade, AnswerDraft])
def test_no_keyword_outside_the_common_subset_survives(model: type) -> None:
    schema = to_provider_schema(model)
    for node in _walk(schema):
        # 'title'/'default' may legitimately be *property names*; only check schema nodes
        for key in BANNED & set(node):
            if key in ("title", "default") and isinstance(node.get(key), dict):
                continue
            pytest.fail(f"{model.__name__}: banned keyword {key!r} in {node}")


def test_refs_are_inlined() -> None:
    items = to_provider_schema(AnswerDraft)["properties"]["citations"]["items"]
    assert items["type"] == "object" and set(items["properties"]) == {
        "document_version_id", "chunk_id"}


def test_optional_fields_become_nullable() -> None:
    tool = to_provider_schema(QueryPlan)["properties"]["tool_request"]
    assert tool["type"] == "object" and tool["nullable"] is True
    assert to_provider_schema(EvidenceGrade)["properties"]["rewritten_query"]["nullable"] is True


def test_constraints_and_enums_are_kept() -> None:
    plan = to_provider_schema(QueryPlan)["properties"]
    assert plan["retrieval_query"]["maxLength"] == 500
    assert set(plan["intent"]["enum"]) == {"policy", "subscription", "hybrid", "clarification"}
    grade = to_provider_schema(EvidenceGrade)["properties"]["confidence"]
    assert grade["minimum"] == 0.0 and grade["maximum"] == 1.0
    assert to_provider_schema(AnswerDraft)["properties"]["citations"]["maxItems"] == 8


def test_required_fields_are_kept() -> None:
    assert set(to_provider_schema(QueryPlan)["required"]) >= {
        "intent", "retrieval_query", "needs_retrieval"}
