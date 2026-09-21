"""SQL for the two retrieval branches (Plan §9.1) — invariant #2, release-blocking.

Both branches are built from the SAME `_SCOPE_FROM` / `_SCOPE_WHERE` fragments, so the version and
tier filters cannot drift apart between vector and full-text search. They are applied inside the
query, before ranking/LIMIT — never as a post-filter on a top-k.

Two things beyond the literal text of Plan §9.1 are required for that to hold in practice:

* `dv.document_id = d.id` — the deferrable FK on documents.active_version_id only proves the
  version exists, not that it belongs to this document.
* `SET LOCAL hnsw.iterative_scan = strict_order` on every vector query. Without it the HNSW index
  returns its `ef_search` (default 40) nearest chunks first and the WHERE clause filters *those*,
  so a pile of closer superseded/internal chunks can empty the result even though the SQL is
  "correct". The join on active_version_id is necessary but not sufficient with HNSW.
"""
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.models import EMBEDDING_DIM
from app.db.vectors import to_vector_literal
from app.retrieval.schemas import Evidence, Tier

_SCOPE_FROM = """
  documents d
  JOIN document_versions dv
    ON dv.id = d.active_version_id
   AND dv.status = 'active'
   AND dv.document_id = d.id
  JOIN chunks c
    ON c.document_version_id = dv.id
"""

_SCOPE_WHERE = """
  d.status = 'active'
  AND d.tier = ANY(CAST(:allowed_tiers AS text[]))
"""

_COLUMNS = """
  c.id AS chunk_id,
  c.document_version_id,
  d.id AS document_id,
  d.title,
  dv.version_no,
  c.chunk_index,
  c.text
"""

# Bind params: :allowed_tiers, :query_embedding, :vector_limit.
# Exposed as a constant so tests can EXPLAIN exactly the statement that runs.
VECTOR_SEARCH_SQL = f"""
SELECT {_COLUMNS},
  c.embedding <=> CAST(:query_embedding AS vector) AS distance
FROM {_SCOPE_FROM}
WHERE {_SCOPE_WHERE}
ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
LIMIT :vector_limit
"""

# plainto_tsquery sanitises arbitrary user text (it never parses operators) and ANDs the terms;
# swapping ' & ' for ' | ' turns that into "any term", which is what a natural-language question
# needs. Bind params: :allowed_tiers, :query_text, :fts_limit.
FULLTEXT_SEARCH_SQL = f"""
WITH q AS (
  SELECT replace(plainto_tsquery('simple', :query_text)::text, ' & ', ' | ')::tsquery AS tsq
)
SELECT {_COLUMNS},
  ts_rank(c.text_search, q.tsq) AS rank
FROM q
CROSS JOIN {_SCOPE_FROM}
WHERE {_SCOPE_WHERE}
  AND c.text_search @@ q.tsq
ORDER BY rank DESC, c.id
LIMIT :fts_limit
"""


def tier_values(allowed_tiers: Sequence[Tier]) -> list[str]:
    """Only real Tier members reach SQL; an unknown value raises instead of being passed along."""
    return [Tier(t).value for t in allowed_tiers]


def _to_evidence(row: Mapping[Any, Any], score: float) -> Evidence:
    return Evidence(
        chunk_id=row["chunk_id"],
        document_version_id=row["document_version_id"],
        document_id=row["document_id"],
        title=row["title"],
        version_no=row["version_no"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        score=score,
    )


async def vector_search(
    conn: AsyncConnection,
    *,
    query_embedding: Sequence[float],
    allowed_tiers: Sequence[Tier],
    limit: int,
) -> list[Evidence]:
    """Nearest active-version chunks by cosine distance; score = cosine similarity."""
    if len(query_embedding) != EMBEDDING_DIM:
        raise ValueError(f"query_embedding must have {EMBEDDING_DIM} dimensions")
    tiers = tier_values(allowed_tiers)
    if not tiers:
        return []  # fail closed

    # Must run in the same transaction as the SELECT, on every call (see module docstring).
    await conn.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    rows = (
        await conn.execute(
            text(VECTOR_SEARCH_SQL),
            {
                "allowed_tiers": tiers,
                "query_embedding": to_vector_literal(query_embedding),
                "vector_limit": limit,
            },
        )
    ).mappings().all()
    # The DB already orders by distance; re-sort only to make ties deterministic.
    ordered = sorted(rows, key=lambda r: (r["distance"], str(r["chunk_id"])))
    return [_to_evidence(r, 1.0 - float(r["distance"])) for r in ordered]


async def fulltext_search(
    conn: AsyncConnection,
    *,
    query_text: str,
    allowed_tiers: Sequence[Tier],
    limit: int,
) -> list[Evidence]:
    """Active-version chunks matching ANY query term, best ts_rank first; score = ts_rank."""
    tiers = tier_values(allowed_tiers)
    if not tiers:
        return []  # fail closed

    rows = (
        await conn.execute(
            text(FULLTEXT_SEARCH_SQL),
            {"allowed_tiers": tiers, "query_text": query_text, "fts_limit": limit},
        )
    ).mappings().all()
    return [_to_evidence(r, float(r["rank"])) for r in rows]
