"""Builds a GraphRuntimeContext around fakes so the graph runs with no DB and no network."""
from collections.abc import Sequence

from app.ai.chat.fake import FakeChatModelClient
from app.ai.embeddings.fake import FakeEmbeddingClient
from app.ai.prompts.loader import load_prompts
from app.ai.recorder import InMemoryModelCallRecorder
from app.ai.registry import ModelRegistry
from app.auth.policies import AuthorizationContext
from app.graph.result import TurnResult
from app.graph.runner import execute_turn
from app.graph.runtime import GraphRuntimeContext, Phase
from app.retrieval.schemas import Evidence, Tier
from tests.unit.ai.helpers import FakeClock, make_budget


class Harness:
    def __init__(  # type: ignore[no-untyped-def]
        self, plan=(), grade=(), answer=(), evidence: Sequence[Evidence] = (), *,
        retrievals: Sequence[Sequence[Evidence]] | None = None,
        tier: Tier = Tier.GENERAL, planner=None, grader=None, answerer=None,
    ) -> None:
        self.clock = FakeClock()
        self.planner = planner or FakeChatModelClient(list(plan))
        self.grader = grader or FakeChatModelClient(list(grade))
        self.answerer = answerer or FakeChatModelClient(list(answer))
        self.embeddings = FakeEmbeddingClient()
        self.recorder = InMemoryModelCallRecorder()
        self._retrievals = [list(r) for r in retrievals] if retrievals else [list(evidence)]
        self.retrieve_calls: list[tuple[str, int, tuple[Tier, ...]]] = []
        self.phases: list[Phase] = []
        self.budget = make_budget(self.clock)
        self.tier = tier

    async def _retrieve(self, query, embedding, allowed_tiers):  # type: ignore[no-untyped-def]
        self.retrieve_calls.append((query, len(embedding), tuple(allowed_tiers)))
        i = min(len(self.retrieve_calls), len(self._retrievals)) - 1
        return self._retrievals[i]

    async def _on_phase(self, phase: Phase) -> None:
        self.phases.append(phase)

    def ctx(self) -> GraphRuntimeContext:
        return GraphRuntimeContext(
            request_id="test",
            auth=AuthorizationContext(user_id="u1", tier=self.tier),
            retriever=self._retrieve,
            models=ModelRegistry(self.planner, self.grader, self.answerer, self.embeddings),
            recorder=self.recorder,
            prompts=load_prompts("v1"),
            budget=self.budget,
            embedding_model="fake-embed",
            chat_timeout_seconds=20.0,
            on_phase=self._on_phase,
        )

    async def run(
        self, question: str = "How long do I have to ask for a refund?", **kw: float
    ) -> TurnResult:
        return await execute_turn(question, self.ctx(), **kw)

    @property
    def chat_calls(self) -> int:
        return len(self.planner.calls) + len(self.grader.calls) + len(self.answerer.calls)
