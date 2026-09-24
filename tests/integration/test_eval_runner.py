"""Eval runner (Plan §16.4, Week 8).

Two things this file proves: (1) the scoring arithmetic (recall@k, citation precision/recall) is
correct on known inputs, seeded directly (not through real ingestion, mirroring
tests/integration/conftest.py's own stated reason for that pattern: retrieval/scoring tests should
not depend on the ingestion code they are meant to be independent of) -- and, in the same test, the
guardrail that answer_correctness/groundedness are NEVER populated by the runner (CLAUDE.md:
"you build the runner, the human owns the gold answers"; no LLM-judge, per the user's explicit
choice -- this must fail if anyone later adds one). (2) The real synthetic corpus + questions.json
actually run end to end in --fake mode without error, which is also the only thing that validates
this repo's hand-authored chunk_index values are correct.
"""
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.chat.fake import FakeChatModelClient
from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.prompts.loader import load_prompts
from app.ai.registry import ModelRegistry, build_fake_registry
from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from app.db.models import EMBEDDING_DIM
from app.eval.report import summarize, write_report
from app.eval.runner import resolve_expected_chunks, run_eval
from app.eval.schema import ExpectedChunk, GoldenQuestion, load_golden_questions
from app.ingestion.fake_embedder import fake_embed
from app.ingestion.parsing import discover_documents, parse_document
from app.ingestion.service import ingest_document
from app.retrieval.schemas import Tier
from app.tools.fake import FakeSubscriptionToolClient
from app.tools.subscription import SubscriptionSnapshot
from tests.integration.conftest import make_chunk, make_document, make_version, point_active

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_ROOT = _REPO_ROOT / "app" / "eval" / "golden" / "corpus"
_QUESTIONS_PATH = _REPO_ROOT / "app" / "eval" / "golden" / "questions.json"


async def test_scoring_arithmetic_is_correct_and_human_columns_stay_blank(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    with db_engine.begin() as conn:
        doc = make_document(conn, "refund.md", tier="general")
        version = make_version(conn, doc, 1, "active", content="Refunds within 14 days.")
        point_active(conn, doc, version)
        matching = make_chunk(conn, version, 0, "Refunds within 14 days.", fake_embed(["a"])[0])
        # A second, unrelated chunk the model must NOT cite -- proves precision penalises an
        # over-broad citation, not just that a correct one was found.
        make_chunk(conn, version, 1, "Unrelated content.", fake_embed(["b"])[0])

    question = GoldenQuestion(
        id="q1", question="refund window", tier=Tier.GENERAL, expected_type="answered",
        expected_chunks=(ExpectedChunk("refund.md", 0), ExpectedChunk("refund.md", 1)),
        expected_citations=(ExpectedChunk("refund.md", 0),),
    )
    plan = QueryPlan(intent="policy", retrieval_query="refund window", needs_retrieval=True)
    grade = EvidenceGrade(sufficient=True, confidence=0.9, reason="fine")
    draft = AnswerDraft(
        status="answered", answer="14 days.",
        citations=[CitationRef(document_version_id=version, chunk_id=matching)],
    )
    models = ModelRegistry(
        FakeChatModelClient([plan]), FakeChatModelClient([grade]), FakeChatModelClient([draft]),
        FakeEmbeddingClient(),
    )

    results = await run_eval(
        async_engine, models, load_prompts("v1"), FakeSubscriptionToolClient(), [question],
        embedding_model="fake", chat_timeout_seconds=20.0, tool_timeout_seconds=5.0,
    )

    assert len(results) == 1
    r = results[0]
    assert r.status_matches_expected is True
    assert r.retrieval_recall_at_k == 1.0  # both expected chunks were retrieved
    assert r.citation_precision == 1.0  # 1 citation, 1 of it expected
    assert r.citation_recall == 1.0  # 1 of the 1 expected citation was made
    assert r.resolution_errors == ()
    assert r.exceeds_call_budget is False
    assert r.tool_routing_correct is True  # no tool expected, none proposed

    # The guardrail: never populated by the runner, regardless of how the turn went.
    assert r.answer_correctness == ""
    assert r.groundedness == ""

    summary = summarize(results)
    assert summary["question_count"] == 1
    assert summary["status_match_rate"] == 1.0


async def test_an_unresolvable_expected_chunk_is_reported_not_silently_dropped(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    """A golden question referencing a chunk that was never ingested (or references a stale
    chunk_index) must show up as a resolution error, not silently score as a miss."""
    question = GoldenQuestion(
        id="q1", question="does not matter", tier=Tier.GENERAL,
        expected_type="insufficient_evidence",
        expected_chunks=(ExpectedChunk("never-ingested.md", 0),),
    )
    plan = QueryPlan(intent="policy", retrieval_query="x", needs_retrieval=True)
    models = ModelRegistry(
        FakeChatModelClient([plan]), FakeChatModelClient([]), FakeChatModelClient([]),
        FakeEmbeddingClient(),
    )

    results = await run_eval(
        async_engine, models, load_prompts("v1"), FakeSubscriptionToolClient(), [question],
        embedding_model="fake", chat_timeout_seconds=20.0, tool_timeout_seconds=5.0,
    )

    assert len(results[0].resolution_errors) == 1
    assert "never-ingested.md" in results[0].resolution_errors[0]
    assert results[0].retrieval_recall_at_k is None  # nothing resolvable to recall against


def _ingest_real_corpus(engine: Engine) -> None:
    for path in discover_documents(_CORPUS_ROOT):
        parsed = parse_document(path, _CORPUS_ROOT)
        ingest_document(
            engine,
            external_id=parsed.external_id, title=parsed.title, tier=parsed.tier,
            content=parsed.content, embed=fake_embed, embedding_model="fake-shake256-eval",
        )


async def test_the_real_corpus_and_questions_run_end_to_end_in_fake_mode(
    db_engine: Engine, async_engine: AsyncEngine, tmp_path: Path
) -> None:
    """Smoke test: also the only thing that validates this repo's hand-authored questions.json
    chunk_index values actually resolve against the real chunker's output (app/ingestion/
    chunking.py)."""
    _ingest_real_corpus(db_engine)
    questions = load_golden_questions(_QUESTIONS_PATH)
    tool_client = FakeSubscriptionToolClient(
        tuple(
            SubscriptionSnapshot(
                subscription_id="sub_1", customer_id="eval", status="active", plan_name="Pro",
                current_period_end="2026-12-31T00:00:00Z", observed_at="2026-09-23T00:00:00Z",
            )
            for _ in range(5)
        )
    )

    results = await run_eval(
        async_engine, build_fake_registry(), load_prompts("v1"), tool_client, questions,
        embedding_model="fake-shake256-eval", chat_timeout_seconds=20.0, tool_timeout_seconds=5.0,
    )

    assert len(results) == len(questions)
    write_report(results, tmp_path)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert all(r.answer_correctness == "" and r.groundedness == "" for r in results)

    by_id = {r.question.id: r for r in results}
    # Every hand-authored expected_chunks/expected_citations reference must resolve against the
    # real corpus -- a resolution error here means questions.json and the corpus have drifted.
    assert all(r.resolution_errors == () for r in results), [
        (r.question.id, r.resolution_errors) for r in results if r.resolution_errors
    ]

    # internal-01/02 (Plan §16.3, invariant #2) check the HARD guarantee directly -- that the
    # specific internal-only fraud-thresholds chunk never reaches a general-tier turn's evidence
    # -- rather than the turn's final status. `demo_responder` (build_fake_registry()) cannot
    # judge topical relevance: it grades "sufficient" from `bool(evidence)` alone (see
    # app/ai/chat/fake.py), so a general-tier query about a fraud threshold still gets *some*
    # (irrelevant) general-tier evidence back from vector search -- which has no relevance floor
    # -- and confidently "answers" from it. A real model is expected to notice the evidence does
    # not address the question and grade it insufficient instead; that is a model-quality
    # question for a human to review in `summary.md`, not something --fake mode can prove either
    # way.
    fraud_chunks, _ = await resolve_expected_chunks(
        async_engine, (ExpectedChunk("fraud-thresholds.md", 0),)
    )
    fraud_chunk_id = next(iter(fraud_chunks.values()))[1]  # (version_id, chunk_id) -> chunk_id
    assert fraud_chunk_id in {str(e.chunk_id) for e in by_id["internal-03"].turn_result.evidence}
    for qid in ("internal-01", "internal-02"):
        seen_chunk_ids = {str(e.chunk_id) for e in by_id[qid].turn_result.evidence}
        assert fraud_chunk_id not in seen_chunk_ids, (
            f"{qid} (general tier) must never see the internal-only fraud-thresholds chunk"
        )
    assert by_id["internal-03"].turn_result.status == "answered"


async def test_embedding_dim_constant_matches_the_test_corpus_fixture() -> None:
    """Sanity guard: if EMBEDDING_DIM ever changes, this file's hermetic fixtures must be revisited
    (fake_embed already tracks it; this just documents the coupling)."""
    assert len(fake_embed(["x"])[0]) == EMBEDDING_DIM
