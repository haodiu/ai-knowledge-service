"""Week 3: ONE sequential turn — plan -> retrieve -> grade -> generate -> validate.

This is deliberately NOT a graph. It has no loops, no rewrite, no retries and no call counter
(at most 3 chat calls by construction); Week 4 replaces it with LangGraph nodes that call the same
`roles.py` services and add the bounded rewrite, MAX_GENERATIVE_LLM_CALLS and the timeout.

Invariant #3/#8: an AnswerDraft is only ever surfaced through `TurnResult.answer` after
`validate_citations` passed. On every other path the draft is dropped, not carried along.
"""
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from app.ai.citations import validate_citations
from app.ai.errors import ModelError, ModelRateLimited, StructuredOutputError
from app.ai.prompts.loader import Prompts
from app.ai.recorder import ModelCallRecorder
from app.ai.registry import ModelRegistry
from app.ai.roles import generate_answer, grade_evidence, plan_query
from app.ai.schemas import CitationRef, EvidenceGrade, QueryPlan, ToolRequest
from app.retrieval.schemas import Evidence

TurnStatus = Literal[
    "answered", "clarification", "insufficient_evidence", "temporarily_unavailable", "blocked"
]
Retriever = Callable[[str, Sequence[float]], Awaitable[Sequence[Evidence]]]


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


async def run_turn(
    *,
    question: str,
    models: ModelRegistry,
    prompts: Prompts,
    recorder: ModelCallRecorder,
    retrieve: Retriever,
    embedding_model: str,
    timeout_seconds: float,
) -> TurnResult:
    plan: QueryPlan | None = None
    grade: EvidenceGrade | None = None
    evidence: tuple[Evidence, ...] = ()
    attempts = 0
    try:
        plan = await plan_query(
            models, recorder, prompts, question, timeout_seconds=timeout_seconds
        )
        proposed = plan.tool_request

        def done(status: TurnStatus, detail: str, **extra: object) -> TurnResult:
            return TurnResult(
                status, plan=plan, grade=grade, evidence=evidence, proposed_tool=proposed,
                detail=detail, retrieval_attempts=attempts, **extra,  # type: ignore[arg-type]
            )

        if plan.intent == "clarification":
            return done("clarification", "planner_clarification",
                        clarification_question=plan.clarification_question)
        if not plan.needs_retrieval:
            return done("insufficient_evidence", "no_retrieval_needed")

        embedded = await models.embeddings.embed(
            [plan.retrieval_query], model_version=embedding_model, kind="query"
        )
        attempts = 1
        evidence = tuple(await retrieve(plan.retrieval_query, embedded.vectors[0]))
        if not evidence:  # nothing to grade: no need to spend a call (Plan §11.2)
            return done("insufficient_evidence", "no_evidence")

        grade = await grade_evidence(
            models, recorder, prompts, question, evidence, timeout_seconds=timeout_seconds
        )
        if not grade.sufficient:  # Week 3: report it; the one bounded rewrite is Week 4
            return done("insufficient_evidence", "grader_insufficient")

        draft = await generate_answer(
            models, recorder, prompts, question, evidence, timeout_seconds=timeout_seconds
        )
        if draft.status == "insufficient_evidence":
            return done("insufficient_evidence", "generator_insufficient")
        if not validate_citations(draft, evidence).valid:
            return done("blocked", "invalid_citations")  # the draft is dropped, never surfaced
        if draft.status == "clarification":
            return done("clarification", "generator_clarification",
                        clarification_question=draft.clarification_question)
        return done("answered", "ok", answer=draft.answer, citations=tuple(draft.citations))
    except StructuredOutputError as exc:
        return _failed("blocked", exc.code, plan, grade, evidence, attempts)
    except ModelRateLimited as exc:
        return _failed("temporarily_unavailable", exc.code, plan, grade, evidence, attempts,
                       retry_after_seconds=exc.retry_after_seconds)
    except ModelError as exc:
        return _failed("temporarily_unavailable", exc.code, plan, grade, evidence, attempts)


def _failed(
    status: TurnStatus,
    detail: str,
    plan: QueryPlan | None,
    grade: EvidenceGrade | None,
    evidence: tuple[Evidence, ...],
    attempts: int,
    *,
    retry_after_seconds: float | None = None,
) -> TurnResult:
    return TurnResult(
        status, plan=plan, grade=grade, evidence=evidence,
        proposed_tool=plan.tool_request if plan else None,
        retry_after_seconds=retry_after_seconds, detail=detail, retrieval_attempts=attempts,
    )
