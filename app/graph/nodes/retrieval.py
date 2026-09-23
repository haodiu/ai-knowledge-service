"""retrieve_documents (deterministic): embed the query, search, merge into the evidence set."""
from typing import Any

from app.ai.errors import ModelError
from app.ai.roles import embed_query
from app.graph.limits import MAX_RETRIEVAL_ATTEMPTS
from app.graph.nodes._errors import to_turn_error
from app.graph.nodes.evidence import build_evidence, evidence_key
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState, TurnError
from app.tools.subscription import is_tool_evidence


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
    # Tool evidence (if any -- only reachable on the second pass of the bounded rewrite loop,
    # since MAX_TOOL_CALLS=1 and the tool always runs before the first grade) is split out and
    # re-merged as `new` so it stays pinned ahead of the cap, exactly like it was on the turn the
    # tool actually ran. Left in `prior`/`existing`, a full page of freshly retrieved chunks could
    # silently evict it -- build_evidence ranks ALL of `new` ahead of ALL of `existing`, not just
    # by score (see app/tools/subscription.py::is_tool_evidence).
    pinned = [e for e in prior if is_tool_evidence(e)]
    doc_prior = [e for e in prior if not is_tool_evidence(e)]
    merged_docs = build_evidence(doc_prior, found)
    merged = build_evidence(merged_docs, pinned) if pinned else merged_docs
    new_count = len({evidence_key(e) for e in merged} - {evidence_key(e) for e in prior})
    return {"retrieval_attempts": attempts + 1, "evidence": merged, "new_evidence_count": new_count}
