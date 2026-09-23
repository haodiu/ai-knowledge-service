"""Request-scoped runtime dependencies (Plan §8.1). Never part of graph state."""
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from app.ai.budget import TurnBudget
from app.ai.prompts.loader import Prompts
from app.ai.recorder import ModelCallRecorder
from app.ai.registry import ModelRegistry
from app.auth.policies import AuthorizationContext
from app.retrieval.schemas import Evidence, Tier
from app.tools.recorder import ToolCallRecorder
from app.tools.subscription import SubscriptionToolClient

# The only events that may leave the graph before validation (invariant #8). They carry no text.
Phase = Literal["planning", "retrieving", "grading", "generating", "calling_tool"]
PhaseCallback = Callable[[Phase], Awaitable[None]]
# (retrieval_query, query_embedding, allowed_tiers) -> evidence. The tiers are handed in by the
# node from `ctx.auth`, so a retriever never has to trust anything the graph state contains.
Retriever = Callable[[str, Sequence[float], Sequence[Tier]], Awaitable[Sequence[Evidence]]]


@dataclass(frozen=True)
class GraphRuntimeContext:
    request_id: str
    auth: AuthorizationContext
    retriever: Retriever
    models: ModelRegistry
    recorder: ModelCallRecorder
    prompts: Prompts
    budget: TurnBudget
    embedding_model: str
    chat_timeout_seconds: float
    # None when no Payment/Subscription host is configured (Plan §18 Tuan 7: none exists yet) --
    # unlike `models`, not every turn needs the tool, so this must not force every request to fail
    # (app/api/dependencies.py::get_tool_client never raises; app/graph/nodes/tools.py is where a
    # missing client becomes a real, turn-scoped `temporarily_unavailable`, only for a plan that
    # actually proposes a tool call).
    tool_client: SubscriptionToolClient | None
    tool_recorder: ToolCallRecorder
    tool_timeout_seconds: float
    on_phase: PhaseCallback | None = None

    async def emit(self, phase: Phase) -> None:
        if self.on_phase is not None:
            await self.on_phase(phase)
