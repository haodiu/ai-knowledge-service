"""Structured LLM outputs (Plan §11.2). Everything here is validated before it has any effect.

Schemas check *shape and internal consistency* only. Anything that needs outside facts (is this
citation in the evidence set? is this tool authorised for this user?) is decided by deterministic
application code (`citations.py`, Week 7 tool gate) — never by the model, never by these schemas.
"""
from typing import Annotated, Literal, Self, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

_Strict = ConfigDict(extra="forbid")

Text500 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Id64 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

ToolName = Literal["get_subscription"]
# The allowlist IS the Literal: adding a tool means adding it here, in code review (invariant #11).
TOOL_ALLOWLIST: frozenset[str] = frozenset(get_args(ToolName))


class GetSubscriptionArgs(BaseModel):
    """Plan §10: additionalProperties=false, exactly one of subscription_id / customer_id."""

    model_config = _Strict

    subscription_id: Id64 | None = None
    customer_id: Id64 | None = None

    @model_validator(mode="after")
    def _exactly_one_identifier(self) -> Self:
        if (self.subscription_id is None) == (self.customer_id is None):
            raise ValueError("exactly one of subscription_id / customer_id is required")
        return self


class ToolRequest(BaseModel):
    """A *proposal*. Nothing executes it in Week 3; Week 7 gates it on allowlist + authorization."""

    model_config = _Strict

    name: ToolName
    arguments: GetSubscriptionArgs


class QueryPlan(BaseModel):
    model_config = _Strict

    intent: Literal["policy", "subscription", "hybrid", "clarification"]
    retrieval_query: Text500
    needs_retrieval: bool
    tool_request: ToolRequest | None = None
    clarification_question: Text500 | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.intent == "policy":
            if self.tool_request is not None or not self.needs_retrieval:
                raise ValueError("policy intent retrieves documents and proposes no tool")
        elif self.intent == "subscription":
            if self.tool_request is None:
                raise ValueError("subscription intent needs a tool_request")
        elif self.intent == "hybrid":
            if self.tool_request is None or not self.needs_retrieval:
                raise ValueError("hybrid intent needs both retrieval and a tool_request")
        else:  # clarification
            if self.clarification_question is None:
                raise ValueError("clarification intent needs a clarification_question")
            if self.needs_retrieval or self.tool_request is not None:
                raise ValueError("clarification intent neither retrieves nor proposes a tool")
        return self


class EvidenceGrade(BaseModel):
    model_config = _Strict

    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)  # NaN/inf fail these comparisons too
    reason: str = Field(max_length=500)
    missing_information: list[Annotated[str, StringConstraints(max_length=200)]] = Field(
        default_factory=list, max_length=5
    )
    # Week 3 only parses this; using it (the one bounded rewrite) is the Week 4 graph's job.
    rewritten_query: Text500 | None = None


class CitationRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_version_id: UUID
    chunk_id: UUID


class AnswerDraft(BaseModel):
    model_config = _Strict

    status: Literal["answered", "clarification", "insufficient_evidence"]
    answer: str | None = None
    clarification_question: Text500 | None = None
    citations: list[CitationRef] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.status == "answered":
            if self.answer is None or not self.answer.strip():
                raise ValueError("an answered draft needs answer text")
            if not self.citations:
                raise ValueError("an answered draft needs at least one citation")
        else:
            if self.citations:
                raise ValueError(f"a {self.status} draft must not carry citations")
            if self.status == "clarification" and self.clarification_question is None:
                raise ValueError("a clarification draft needs a clarification_question")
        return self
