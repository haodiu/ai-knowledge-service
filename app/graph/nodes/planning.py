"""plan_query (LLM #1). Runs BEFORE retrieval: the planner never sees retrieved text."""
from typing import Any

from app.ai.errors import ModelError
from app.ai.roles import plan_query
from app.graph.nodes._errors import to_turn_error
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState


async def plan_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    await ctx.emit("planning")
    try:
        plan = await plan_query(
            ctx.models, ctx.recorder, ctx.prompts, state["normalized_question"],
            timeout_seconds=ctx.chat_timeout_seconds, budget=ctx.budget,
        )
    except ModelError as exc:
        return {"error": to_turn_error(exc)}
    # tool_request is recorded as a proposal only; no node executes it (Week 7, invariant #11)
    return {"plan": plan, "proposed_tool": plan.tool_request,
            "retrieval_query": plan.retrieval_query}
