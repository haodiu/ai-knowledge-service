"""Conversation history (Plan §11.5, Week 6). `app/graph/state.py` used to hardcode
`recent_turns: []` until this landed -- these tests cover both the repository query and the
end-to-end wiring into the planner's prompt.
"""
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.chat.fake import FakeChatModelClient
from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.prompts.loader import load_prompts
from app.ai.registry import ModelRegistry
from app.ai.schemas import AnswerDraft, EvidenceGrade, QueryPlan
from app.auth.policies import AuthorizationContext
from app.db import repositories
from app.graph.runner import run_turn
from app.retrieval.schemas import Tier
from app.tools.fake import FakeSubscriptionToolClient

pytestmark = pytest.mark.integration


async def test_get_recent_turns_returns_only_answered_turns_oldest_first(
    async_engine: AsyncEngine,
) -> None:
    created = await repositories.create_conversation(async_engine, "u")
    turn_ids = []
    for i in range(3):
        t = await repositories.add_turn(async_engine, created, f"question {i}")
        turn_ids.append(t)
        await repositories.finish_turn(
            async_engine, t, graph_status="answered", answer=f"answer {i}", sources=[],
            retrieval_attempts=1, latency_ms=1,
        )
    running = await repositories.add_turn(async_engine, created, "not answered yet")  # noqa: F841

    history = await repositories.get_recent_turns(
        async_engine, created, limit=4, max_chars_per_turn=500
    )

    assert len(history) == 3  # the running turn is excluded
    assert history[0].startswith("Q: question 0")  # oldest first
    assert history[-1].startswith("Q: question 2")


async def test_get_recent_turns_respects_the_limit(async_engine: AsyncEngine) -> None:
    created = await repositories.create_conversation(async_engine, "u")
    for i in range(6):
        t = await repositories.add_turn(async_engine, created, f"q{i}")
        await repositories.finish_turn(
            async_engine, t, graph_status="answered", answer=f"a{i}", sources=[],
            retrieval_attempts=1, latency_ms=1,
        )

    history = await repositories.get_recent_turns(
        async_engine, created, limit=4, max_chars_per_turn=500
    )

    assert len(history) == 4
    assert [h.split("\n")[0] for h in history] == ["Q: q2", "Q: q3", "Q: q4", "Q: q5"]


async def test_get_recent_turns_truncates_long_turns(async_engine: AsyncEngine) -> None:
    created = await repositories.create_conversation(async_engine, "u")
    t = await repositories.add_turn(async_engine, created, "q" * 1000)
    await repositories.finish_turn(
        async_engine, t, graph_status="answered", answer="a" * 1000, sources=[],
        retrieval_attempts=1, latency_ms=1,
    )

    (history_entry,) = await repositories.get_recent_turns(
        async_engine, created, limit=4, max_chars_per_turn=50
    )

    assert len(history_entry) < 150  # well under the untruncated ~2000 chars


async def test_a_fresh_conversation_has_no_history(async_engine: AsyncEngine) -> None:
    created = await repositories.create_conversation(async_engine, "u")
    history = await repositories.get_recent_turns(
        async_engine, created, limit=4, max_chars_per_turn=500
    )
    assert history == []


def _fake_models() -> tuple[ModelRegistry, FakeChatModelClient]:
    # intent="clarification" ends the graph right after planning -- no retrieval/grade/answer
    # calls needed, keeping this test focused on what the planner's prompt contained.
    planner = FakeChatModelClient([
        QueryPlan(
            intent="clarification", retrieval_query="n/a", needs_retrieval=False,
            clarification_question="Which plan?",
        )
    ])
    grader = FakeChatModelClient([EvidenceGrade(sufficient=False, confidence=0.1, reason="none")])
    answerer = FakeChatModelClient([AnswerDraft(status="insufficient_evidence")])
    return ModelRegistry(planner, grader, answerer, FakeEmbeddingClient()), planner


async def test_run_turn_feeds_prior_answered_turns_to_the_planner(
    async_engine: AsyncEngine,
) -> None:
    """End-to-end: a second turn in the same conversation sees the first turn's Q/A in the
    planner's prompt (app.graph.nodes.understand -> run_graph -> plan_node)."""
    created = await repositories.create_conversation(async_engine, "u")
    first = await repositories.add_turn(async_engine, created, "What is the refund window?")
    await repositories.finish_turn(
        async_engine, first, graph_status="answered", answer="14 days.", sources=[],
        retrieval_attempts=1, latency_ms=1,
    )

    second = await repositories.add_turn(async_engine, created, "What about annual plans?")
    models, planner = _fake_models()
    await run_turn(
        engine=async_engine, conversation_id=created, turn_id=second,
        question="What about annual plans?",
        auth=AuthorizationContext(user_id="u", tier=Tier.GENERAL),
        models=models, prompts=load_prompts("v1"), embedding_model="fake",
        chat_timeout_seconds=20.0,
        tool_client=FakeSubscriptionToolClient(), tool_timeout_seconds=5.0,
    )

    planner_prompt = "\n".join(m.content for m in planner.calls[0].messages)
    assert "What is the refund window?" in planner_prompt
    assert "14 days." in planner_prompt


async def test_run_turn_does_not_see_its_own_turn_as_history(
    async_engine: AsyncEngine,
) -> None:
    """The turn row is inserted (status='running') before run_turn() executes, so it must not
    leak into its own recent_turns."""
    created = await repositories.create_conversation(async_engine, "u")
    turn_id = await repositories.add_turn(async_engine, created, "self-referential question")
    models, planner = _fake_models()

    await run_turn(
        engine=async_engine, conversation_id=created, turn_id=turn_id,
        question="self-referential question",
        auth=AuthorizationContext(user_id="u", tier=Tier.GENERAL),
        models=models, prompts=load_prompts("v1"), embedding_model="fake",
        chat_timeout_seconds=20.0,
        tool_client=FakeSubscriptionToolClient(), tool_timeout_seconds=5.0,
    )

    planner_prompt = "\n".join(m.content for m in planner.calls[0].messages)
    assert "Recent conversation" not in planner_prompt
