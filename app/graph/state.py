"""Graph state (Plan §8.1): serialisable data only.

No JWT, token, model client, DB session, retriever, budget or authorization object — those are
request-scoped runtime dependencies in GraphRuntimeContext. `tier` in particular is absent from
state on purpose: nothing a node (or a model) writes to state can change who is asking.
"""
from dataclasses import dataclass
from typing import Literal, TypedDict

from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan, ToolRequest
from app.retrieval.schemas import Evidence, SourceSnapshot


@dataclass(frozen=True)
class TurnError:
    """A failure a node caught. Routers send any state with an error straight to the fallback."""

    kind: Literal["blocked", "unavailable", "clarification"]  # clarification: tool 409 (Week 7)
    code: str
    retry_after_seconds: float | None = None
    clarification_question: str | None = None  # only set when kind == "clarification"


class RAGState(TypedDict, total=False):
    question: str
    normalized_question: str
    recent_turns: list[str]  # always empty until conversation history lands (Week 6)

    plan: QueryPlan | None
    proposed_tool: ToolRequest | None  # the planner's proposal, whether or not it gets executed
    tool_executed: bool  # was the tool actually called this turn (any outcome), set by tool_node

    retrieval_query: str
    retrieval_attempts: int
    evidence: list[Evidence]
    new_evidence_count: int  # chunks the latest retrieval added to `evidence`

    grade: EvidenceGrade | None
    draft: AnswerDraft | None  # unvalidated; never copied to `answer` except by validate_node
    citations_valid: bool | None

    error: TurnError | None
    status: str
    detail: str
    answer: str | None
    clarification_question: str | None
    citations: list[CitationRef]
    source_snapshots: list[SourceSnapshot]
    retry_after_seconds: float | None
