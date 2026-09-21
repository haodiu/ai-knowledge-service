"""generate_answer (LLM #3). Only reached when the grader said the evidence is sufficient."""
from typing import Any

from app.ai.errors import ModelError
from app.ai.roles import generate_answer
from app.graph.nodes._errors import to_turn_error
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState


async def generate_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    await ctx.emit("generating")
    try:
        draft = await generate_answer(
            ctx.models, ctx.recorder, ctx.prompts, state["normalized_question"],
            state["evidence"], timeout_seconds=ctx.chat_timeout_seconds, budget=ctx.budget,
        )
    except ModelError as exc:
        return {"error": to_turn_error(exc)}
    return {"draft": draft}  # unvalidated: it is NOT the answer yet
