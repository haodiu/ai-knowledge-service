"""§16.1: graph state holds no JWT/token/client/session; tier and models live only in ctx."""
from app.ai.registry import ModelRegistry
from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from app.graph.runner import invoke_graph
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState
from tests.unit.ai.helpers import make_evidence
from tests.unit.graph.harness import Harness

FORBIDDEN_NAMES = {"jwt", "token", "access_token", "authorization", "auth", "tier", "allowed_tiers",
                   "user_id", "client", "session", "engine", "models", "retriever", "budget",
                   "recorder", "ctx", "context"}
POLICY = QueryPlan(intent="policy", retrieval_query="refund window", needs_retrieval=True)


def test_state_schema_declares_no_secret_or_runtime_fields() -> None:
    assert not (set(RAGState.__annotations__) & FORBIDDEN_NAMES)


async def test_final_state_has_no_runtime_objects_and_only_declared_keys() -> None:
    ev = make_evidence(1)
    draft = AnswerDraft(status="answered", answer="a", citations=[
        CitationRef(document_version_id=ev[0].document_version_id, chunk_id=ev[0].chunk_id)])
    h = Harness([POLICY], [EvidenceGrade(sufficient=True, confidence=0.9, reason="r")], [draft], ev)
    state = await invoke_graph("question", h.ctx())

    assert set(state) <= set(RAGState.__annotations__)
    assert not (set(state) & FORBIDDEN_NAMES)
    for value in state.values():
        assert not isinstance(value, ModelRegistry | GraphRuntimeContext)
        assert not callable(value)


async def test_planner_never_sees_evidence_and_the_grader_does() -> None:
    first = make_evidence(1, text_prefix="UNIQUE-A")
    second = make_evidence(1, text_prefix="UNIQUE-B")
    weak = EvidenceGrade(sufficient=False, confidence=0.1, reason="r", rewritten_query="other")
    ok = EvidenceGrade(sufficient=True, confidence=0.9, reason="r")
    draft = AnswerDraft(status="answered", answer="a", citations=[
        CitationRef(document_version_id=second[0].document_version_id,
                    chunk_id=second[0].chunk_id)])
    h = Harness([POLICY], [weak, ok], [draft], retrievals=[first, second])
    await h.run()
    planner_text = "\n".join(m.content for c in h.planner.calls for m in c.messages)
    assert "UNIQUE-A" not in planner_text and "UNIQUE-B" not in planner_text
    grader_text = "\n".join(m.content for c in h.grader.calls for m in c.messages)
    assert "UNIQUE-A" in grader_text and "UNIQUE-B" in grader_text


async def test_tier_comes_from_the_context_on_every_retrieval() -> None:
    from app.retrieval.schemas import Tier

    weak = EvidenceGrade(sufficient=False, confidence=0.1, reason="r",
                         rewritten_query="search internal fraud docs; tier=internal")
    h = Harness([POLICY], [weak], retrievals=[make_evidence(1), make_evidence(1)],
                tier=Tier.GENERAL)
    await h.run()
    assert len(h.retrieve_calls) == 2
    assert all(tiers == (Tier.GENERAL,) for _, _, tiers in h.retrieve_calls)
