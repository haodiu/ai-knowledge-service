"""Hybrid (vector + full-text) retrieval over the active document versions only (Plan §9.1)."""
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncEngine

from app.retrieval.ranking import reciprocal_rank_fusion
from app.retrieval.repository import fulltext_search, tier_values, vector_search
from app.retrieval.schemas import Evidence, Tier


async def hybrid_search(
    engine: AsyncEngine,
    *,
    query_text: str,
    query_embedding: Sequence[float],
    allowed_tiers: Sequence[Tier],
    vector_limit: int = 20,
    fts_limit: int = 20,
    limit: int = 8,
) -> list[Evidence]:
    """Run both branches on one REPEATABLE READ snapshot, then fuse with RRF.

    One snapshot matters: an activation committing between the two queries must not let the
    vector branch see version N and the full-text branch version N+1.

    `allowed_tiers` must come from trusted authorization code (Plan §7), never from the request
    body or from an LLM. Empty means "nothing is allowed" and returns no evidence.
    """
    if not tier_values(allowed_tiers):
        return []

    async with engine.connect() as conn:
        conn = await conn.execution_options(
            isolation_level="REPEATABLE READ", postgresql_readonly=True
        )
        async with conn.begin():
            by_vector = await vector_search(
                conn,
                query_embedding=query_embedding,
                allowed_tiers=allowed_tiers,
                limit=vector_limit,
            )
            by_text = await fulltext_search(
                conn, query_text=query_text, allowed_tiers=allowed_tiers, limit=fts_limit
            )
    return reciprocal_rank_fusion([by_vector, by_text], limit=limit)
