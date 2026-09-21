"""What a turn returns to its caller. `answer` is only ever set from a validated draft."""
from dataclasses import dataclass
from typing import Literal

from app.ai.schemas import CitationRef, EvidenceGrade, QueryPlan, ToolRequest
from app.retrieval.schemas import Evidence, SourceSnapshot

TurnStatus = Literal[
    "answered", "clarification", "insufficient_evidence", "temporarily_unavailable", "blocked"
]

# --- every value TurnResult.detail may take. Nothing outside this set is ever produced: the
# runner turns an unknown detail into `unexpected_state` and logs it, so a new failure path cannot
# hide in an unnamed branch. Grouped by the status they belong to.
#
# What a FAILING stage (plan / retrieve+embed / grade / generate) ends as — the normalised error
# code of the provider failure becomes the detail:
#   invalid output, after the ONE repair   -> blocked                 / structured_output_invalid
#   429                                     -> temporarily_unavailable / rate_limited (+retry_after)
#   timeout                                 -> temporarily_unavailable / timeout
#   provider down or misconfigured          -> temporarily_unavailable / unavailable
#   no call unit or deadline left           -> temporarily_unavailable / budget_exhausted
#   unusable embedding                      -> temporarily_unavailable / embedding_dimension|invalid
DETAILS_ANSWERED = frozenset({"ok"})
DETAILS_NON_ANSWER = frozenset({
    "planner_clarification", "generator_clarification",           # clarification
    "no_retrieval_needed", "no_evidence", "grader_insufficient",  # insufficient_evidence
    "rewrite_rejected", "rewrite_no_new_evidence", "generator_insufficient",
})
DETAILS_BLOCKED = frozenset({
    "invalid_citations", "invalid_question", "retrieval_attempts_exceeded",
    "structured_output_invalid",
})
DETAILS_UNAVAILABLE = frozenset({
    "model_error", "timeout", "rate_limited", "unavailable", "budget_exhausted",
    "embedding_dimension", "embedding_invalid", "graph_timeout", "persistence_failed",
})
# Graph BUGS: reachable only through an unexpected exception or an impossible state. They are named
# (never silent) but must never be produced by a legitimate failure; tests assert exactly that.
BUG_DETAILS = frozenset({"internal_error", "unexpected_state", "graph_recursion_limit"})

KNOWN_DETAILS: frozenset[str] = (
    DETAILS_ANSWERED | DETAILS_NON_ANSWER | DETAILS_BLOCKED | DETAILS_UNAVAILABLE | BUG_DETAILS
)


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
