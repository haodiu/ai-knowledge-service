"""Every way a stage can fail maps to ONE named (status, detail) — nothing falls into an unnamed
branch. This is the table (also documented in app/graph/result.py and routing.fallback_outcome):

  invalid output (after the one repair)      -> blocked                  / structured_output_invalid
  429                                        -> temporarily_unavailable  / rate_limited
  timeout                                    -> temporarily_unavailable  / timeout
  provider down or misconfigured (5xx/4xx)   -> temporarily_unavailable  / unavailable
  no call budget / deadline left             -> temporarily_unavailable  / budget_exhausted
  embedding wrong size / unusable            -> temporarily_unavailable  / embedding_*
  graph deadline                             -> temporarily_unavailable  / graph_timeout
  answer could not be persisted              -> temporarily_unavailable  / persistence_failed
  unexpected exception / impossible state    -> BUG details (internal_error, unexpected_state)

The original bug (generate failing -> KeyError in validate) surfaced as `internal_error`: named, but
it hid a graph bug behind a legitimate-looking status. The matrix asserts no stage ever does that.
"""
import pytest

from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.errors import (
    EmbeddingDimensionError,
    EmbeddingInvalidError,
    ModelError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from app.graph.result import BUG_DETAILS, KNOWN_DETAILS, TurnResult
from tests.unit.ai.helpers import make_evidence
from tests.unit.graph.harness import Harness

POLICY = QueryPlan(intent="policy", retrieval_query="refund window", needs_retrieval=True)
SUFFICIENT = EvidenceGrade(sufficient=True, confidence=0.9, reason="ok")
BAD = '{"status": "answered"'

FAILURES = {
    "invalid_output": ([BAD, BAD], "blocked", "structured_output_invalid"),
    "503_twice": ([ModelUnavailable("503"), ModelUnavailable("503")],
                  "temporarily_unavailable", "unavailable"),
    "timeout_twice": ([ModelTimeout("t"), ModelTimeout("t")],
                      "temporarily_unavailable", "timeout"),
    "rate_limited": ([ModelRateLimited("429", retry_after_seconds=30.0)],
                     "temporarily_unavailable", "rate_limited"),
    "auth_or_config": ([ModelUnavailable("401", retryable=False)],
                       "temporarily_unavailable", "unavailable"),
}


def _harness(stage: str, script: list) -> tuple[Harness, list]:  # type: ignore[type-arg]
    ev = make_evidence(1)
    answered = AnswerDraft(status="answered", answer="a", citations=[
        CitationRef(document_version_id=ev[0].document_version_id, chunk_id=ev[0].chunk_id)])
    plan, grade, answer = [POLICY], [SUFFICIENT], [answered]
    if stage == "plan":
        plan = script
    elif stage == "grade":
        grade = script
    else:
        answer = script
    return Harness(plan, grade, answer, ev), script


def _assert_clean(result: TurnResult, status: str, detail: str) -> None:
    assert (result.status, result.detail) == (status, detail)
    assert detail in KNOWN_DETAILS and detail not in BUG_DETAILS
    assert result.answer is None and result.citations == () and result.source_snapshots == ()


@pytest.mark.parametrize("stage", ["plan", "grade", "generate"])
@pytest.mark.parametrize("name", list(FAILURES))
async def test_every_stage_failure_maps_to_one_named_detail(stage: str, name: str) -> None:
    script, status, detail = FAILURES[name]
    h, _ = _harness(stage, list(script))
    result = await h.run()

    _assert_clean(result, status, detail)
    assert h.recorder.records[-1].error_code == detail  # model_calls and the turn agree
    assert h.recorder.records[-1].status == "error"
    if name == "rate_limited":
        assert result.retry_after_seconds == 30.0


@pytest.mark.parametrize("stage", ["plan", "grade", "generate"])
async def test_a_stage_that_cannot_get_a_call_unit_is_budget_exhausted(stage: str) -> None:
    h, _ = _harness(stage, [])
    # give the pool exactly the units used BEFORE this stage
    h.budget.max_generative_calls = {"plan": 0, "grade": 1, "generate": 2}[stage]
    result = await h.run()
    _assert_clean(result, "temporarily_unavailable", "budget_exhausted")
    assert h.chat_calls == {"plan": 0, "grade": 1, "generate": 2}[stage]  # no request was made


@pytest.mark.parametrize(
    ("error", "detail"),
    [(ModelRateLimited("429", retry_after_seconds=None), "rate_limited"),
     (ModelUnavailable("down", retryable=False), "unavailable"),
     (EmbeddingDimensionError("768"), "embedding_dimension"),
     (EmbeddingInvalidError("zero"), "embedding_invalid")],
)
async def test_embedding_failures_are_named_and_stop_before_retrieval(
    error: ModelError, detail: str
) -> None:
    h = Harness([POLICY], evidence=make_evidence(1))
    h.embeddings = FakeEmbeddingClient(fail_with=error)
    result = await h.run()
    _assert_clean(result, "temporarily_unavailable", detail)
    assert h.retrieve_calls == []


def test_every_error_code_a_model_error_can_carry_is_a_known_detail() -> None:
    """Adding a ModelError subclass with a new code must also name it in KNOWN_DETAILS."""
    def walk(cls: type[ModelError]):  # type: ignore[no-untyped-def]
        yield cls
        for sub in cls.__subclasses__():
            yield from walk(sub)

    codes = {c.code for c in walk(ModelError)}
    assert codes <= KNOWN_DETAILS, codes - KNOWN_DETAILS


def test_bug_details_are_a_subset_of_the_known_details() -> None:
    assert BUG_DETAILS <= KNOWN_DETAILS
    assert BUG_DETAILS == {"internal_error", "unexpected_state", "graph_recursion_limit"}


@pytest.mark.parametrize(
    ("state", "why"),
    [({"status": "insufficient_evidence", "detail": "made_up_detail"}, "unknown detail"),
     ({"status": "answered", "detail": "no_evidence", "answer": "x"}, "answered without ok"),
     ({"status": "insufficient_evidence", "detail": "ok"}, "ok on a non-answer"),
     ({}, "empty state")],
)
async def test_an_unnamed_or_inconsistent_outcome_fails_closed_as_unexpected_state(
    monkeypatch: pytest.MonkeyPatch, state: dict, why: str  # type: ignore[type-arg]
) -> None:
    from app.graph import runner

    async def fake_run_graph(question, ctx, *, recursion_limit):  # type: ignore[no-untyped-def]
        return state

    monkeypatch.setattr(runner, "run_graph", fake_run_graph)
    result = await Harness().run()
    assert (result.status, result.detail) == ("blocked", "unexpected_state"), why
    assert result.answer is None and result.source_snapshots == ()
