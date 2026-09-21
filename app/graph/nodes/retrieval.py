"""retrieve_documents (deterministic): embed the query, search, merge into the evidence set."""
from typing import Any

from app.ai.errors import ModelError
from app.ai.roles import embed_query
from app.graph.limits import MAX_RETRIEVAL_ATTEMPTS
from app.graph.nodes._errors import to_turn_error
from app.graph.nodes.evidence import build_evidence, evidence_key
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState, TurnError


async def retrieve_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    attempts = state.get("retrieval_attempts", 0)
    if attempts >= MAX_RETRIEVAL_ATTEMPTS:  # second guard behind the router (defence in depth)
        return {"error": TurnError(kind="blocked", code="retrieval_attempts_exceeded")}

    await ctx.emit("retrieving")
    query = state["retrieval_query"]
    try:
        vector = await embed_query(
            ctx.models, ctx.budget, query, model_version=ctx.embedding_model
        )
    except ModelError as exc:
        return {"error": to_turn_error(exc)}

    # Tiers come from the trusted context, never from state or a model (invariant #1).
    found = await ctx.retriever(query, vector, ctx.auth.allowed_tiers)

    prior = state.get("evidence", [])
    merged = build_evidence(prior, found)
    new_count = len({evidence_key(e) for e in merged} - {evidence_key(e) for e in prior})
    return {"retrieval_attempts": attempts + 1, "evidence": merged, "new_evidence_count": new_count}
