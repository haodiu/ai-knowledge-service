"""What a turn returns to its caller. `answer` is only ever set from a validated draft."""
from dataclasses import dataclass
from typing import Literal

from app.ai.schemas import CitationRef, EvidenceGrade, QueryPlan, ToolRequest
from app.retrieval.schemas import Evidence, SourceSnapshot

TurnStatus = Literal[
    "answered", "clarification", "insufficient_evidence", "temporarily_unavailable", "blocked"
]


@dataclass(frozen=True)
class TurnResult:
    status: TurnStatus
    answer: str | None = None
    clarification_question: str | None = None
    citations: tuple[CitationRef, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    plan: QueryPlan | None = None
    grade: EvidenceGrade | None = None
    proposed_tool: ToolRequest | None = None
    tool_executed: bool = False  # always False until Week 7; a proposal is never a call
    retry_after_seconds: float | None = None
    detail: str = ""
    retrieval_attempts: int = 0
    source_snapshots: tuple[SourceSnapshot, ...] = ()
