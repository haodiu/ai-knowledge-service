"""Invariant #2 (release-blocking): retrieval only ever sees documents.active_version_id.

Written BEFORE the implementation (CLAUDE.md, Testing & CI gates). Every state is seeded with
raw SQL so these tests do not depend on the ingestion code. In every scenario the non-active
versions' chunks are *better* matches than the active one (closer vector, more query terms),
so a missing/late version filter shows up as a wrong result, not as a lucky pass.
"""
import math
import random
import uuid
from dataclasses import dataclass

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.db.models import EMBEDDING_DIM
from app.retrieval import repository
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.repository import fulltext_search, vector_search
from app.retrieval.schemas import Evidence, Tier
from tests.integration.conftest import (
    make_chunk,
    make_document,
    make_version,
    point_active,
    unit_vector,
    vec_literal,
)

pytestmark = pytest.mark.integration

QUERY_TEXT = "refund policy"
QUERY_VEC = unit_vector(0)  # e0
BOTH_TIERS = [Tier.GENERAL, Tier.INTERNAL]


@dataclass
class Seeded:
    document_id: uuid.UUID
    active_version: uuid.UUID
    active_chunk: uuid.UUID
    other_chunks: set[uuid.UUID]  # every chunk of a non-active version


def _seed_four_states(
    engine: Engine, *, external_id: str = "policy", tier: str = "general"
) -> Seeded:
    """One document, one version in each status, all coexisting (Plan §5.2)."""
    with engine.begin() as conn:
        doc = make_document(conn, external_id, tier=tier)
        superseded = make_version(conn, doc, 1, "superseded")
        active = make_version(conn, doc, 2, "active")
        building = make_version(conn, doc, 3, "building")
        failed = make_version(conn, doc, 4, "failed")
        point_active(conn, doc, active)

        # Non-active chunks: distance 0 to the query and every query term present.
        others = {
            make_chunk(conn, superseded, 0, "refund policy old thirty days", unit_vector(0)),
            make_chunk(conn, building, 0, "refund policy draft seven days", unit_vector(0)),
            make_chunk(conn, failed, 0, "refund policy failed build", unit_vector(0)),
        }
        # Active chunk: farther vector (cos distance ~0.106) and it is the only one that must win.
        active_chunk = make_chunk(
            conn, active, 0, "refund policy current fourteen days", unit_vector(0, blend=0.5)
        )
    return Seeded(doc, active, active_chunk, others)


def _ids(evidence: list[Evidence]) -> set[uuid.UUID]:
    return {e.chunk_id for e in evidence}


# --- the required minimum: active/building/superseded coexisting ---------------------------


async def test_vector_branch_returns_only_the_active_version(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    seeded = _seed_four_states(db_engine)
    async with async_engine.connect() as conn:
        result = await vector_search(
            conn, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS, limit=10
        )
    assert _ids(result) == {seeded.active_chunk}
    assert result[0].document_version_id == seeded.active_version


async def test_fulltext_branch_returns_only_the_active_version(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    seeded = _seed_four_states(db_engine)
    async with async_engine.connect() as conn:
        result = await fulltext_search(
            conn, query_text=QUERY_TEXT, allowed_tiers=BOTH_TIERS, limit=10
        )
    assert _ids(result) == {seeded.active_chunk}
    assert result[0].document_version_id == seeded.active_version


async def test_hybrid_evidence_only_contains_active_chunks(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    seeded = _seed_four_states(db_engine)
    result = await hybrid_search(
        async_engine, query_text=QUERY_TEXT, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS
    )
    assert _ids(result) == {seeded.active_chunk}
    # The pair used for citation validation (invariant 3) must be the active version's.
    assert {(e.document_version_id, e.chunk_id) for e in result} == {
        (seeded.active_version, seeded.active_chunk)
    }


async def test_document_without_an_active_version_is_invisible(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    with db_engine.begin() as conn:
        doc = make_document(conn, "brand-new")
        building = make_version(conn, doc, 1, "building")
        make_chunk(conn, building, 0, "refund policy draft", unit_vector(0))
    result = await hybrid_search(
        async_engine, query_text=QUERY_TEXT, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS
    )
    assert result == []


async def test_archived_document_is_invisible_even_with_an_active_version(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    with db_engine.begin() as conn:
        doc = make_document(conn, "archived", status="archived")
        active = make_version(conn, doc, 1, "active")
        point_active(conn, doc, active)
        make_chunk(conn, active, 0, "refund policy archived", unit_vector(0))
    result = await hybrid_search(
        async_engine, query_text=QUERY_TEXT, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS
    )
    assert result == []


# --- the pointer alone is not enough: status and ownership are checked too -------------------


async def test_pointer_to_a_non_active_version_is_not_served(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    """documents.active_version_id -> a superseded version (inconsistent data): fail closed."""
    with db_engine.begin() as conn:
        doc = make_document(conn, "inconsistent")
        old = make_version(conn, doc, 1, "superseded")
        point_active(conn, doc, old)
        make_chunk(conn, old, 0, "refund policy old", unit_vector(0))
    async with async_engine.connect() as conn:
        assert (
            await vector_search(conn, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS, limit=10)
            == []
        )
        assert (
            await fulltext_search(conn, query_text=QUERY_TEXT, allowed_tiers=BOTH_TIERS, limit=10)
            == []
        )


async def test_pointer_that_disagrees_with_version_status_serves_nothing(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    """v1 is status=active but documents.active_version_id says v2 (building). The pointer is the
    authority (invariant 2 requires BOTH conditions), so neither version may be served."""
    with db_engine.begin() as conn:
        doc = make_document(conn, "disagree")
        v1 = make_version(conn, doc, 1, "active")
        v2 = make_version(conn, doc, 2, "building")
        make_chunk(conn, v1, 0, "refund policy v1", unit_vector(0))
        make_chunk(conn, v2, 0, "refund policy v2", unit_vector(0))
        point_active(conn, doc, v2)
    async with async_engine.connect() as conn:
        assert (
            await vector_search(conn, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS, limit=10)
            == []
        )
        assert (
            await fulltext_search(conn, query_text=QUERY_TEXT, allowed_tiers=BOTH_TIERS, limit=10)
            == []
        )


async def test_pointer_to_another_documents_version_is_not_served_under_the_wrong_document(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    """The deferrable FK only proves the version exists. Doc A pointing at doc B's active version
    must not surface B's chunk a second time, attributed to A (wrong title/tier)."""
    with db_engine.begin() as conn:
        doc_a = make_document(conn, "doc-a", tier="internal")
        va = make_version(conn, doc_a, 1, "active")
        point_active(conn, doc_a, va)
        doc_b = make_document(conn, "doc-b", tier="general")
        vb = make_version(conn, doc_b, 1, "active")
        point_active(conn, doc_b, vb)
        chunk_b = make_chunk(conn, vb, 0, "refund policy of b", unit_vector(0))
        point_active(conn, doc_a, vb)  # corrupt: A now points at B's version
    async with async_engine.connect() as conn:
        vec = await vector_search(
            conn, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS, limit=10
        )
        fts = await fulltext_search(conn, query_text=QUERY_TEXT, allowed_tiers=BOTH_TIERS, limit=10)
    for result in (vec, fts):
        assert [(e.chunk_id, e.document_id) for e in result] == [(chunk_b, doc_b)]


# --- HNSW: the filter must apply before the ANN candidate cut-off --------------------------


def _noisy_vector(rng: random.Random, noise_norm: float) -> list[float]:
    """e0 plus Gaussian noise of a fixed norm: cosine distance to the query grows with noise_norm.

    Random high-dimensional points give the HNSW graph a realistic, well-connected shape (unlike
    hand-made near-duplicates, which can leave a node unreachable and prove nothing).
    """
    noise = [rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)]
    scale = noise_norm / math.sqrt(sum(x * x for x in noise))
    vec = [x * scale for x in noise]
    vec[0] += 1.0
    return vec


def _seed_many_superseded_closer_than_active(engine: Engine, n: int) -> tuple[uuid.UUID, uuid.UUID]:
    """n superseded chunks all closer to the query than the single active chunk, n > ef_search."""
    with engine.begin() as conn:
        doc = make_document(conn, "hnsw")
        old = make_version(conn, doc, 1, "superseded")
        active = make_version(conn, doc, 2, "active")
        point_active(conn, doc, active)
        # Every superseded chunk is closer to the query (noise norm <= 0.4, cosine distance
        # <= ~0.07) than the single active chunk (noise norm 0.7, distance ~0.16).
        rng = random.Random(20260921)
        conn.execute(
            text(
                "INSERT INTO chunks (document_version_id, chunk_index, text, embedding) "
                "VALUES (:v, :i, :t, CAST(:e AS vector))"
            ),
            [
                {
                    "v": old,
                    "i": i,
                    "t": f"refund policy superseded {i}",
                    "e": vec_literal(_noisy_vector(rng, 0.05 + 0.35 * i / n)),
                }
                for i in range(n)
            ],
        )
        active_chunk = make_chunk(
            conn, active, 0, "refund policy active", _noisy_vector(rng, 0.7)
        )
        conn.execute(text("ANALYZE chunks"))
    return active, active_chunk


async def test_active_chunk_found_when_hnsw_candidates_are_all_superseded(
    db_engine: Engine, test_db_url: str
) -> None:
    """The silent post-filter failure: 200 superseded chunks (> hnsw.ef_search=40) outrank the
    only active one. A plain HNSW scan returns the 40 nearest, all superseded, then the version
    filter empties the result. `SET LOCAL hnsw.iterative_scan = strict_order` must prevent that.

    seqscan and explicit sorts are disabled on this engine so the planner must take the
    pre-ordered HNSW index path (on a tiny table it otherwise prefers an exact join + Sort, which
    would make this test vacuous). The EXPLAIN assertion proves the index really was used.
    """
    _, active_chunk = _seed_many_superseded_closer_than_active(db_engine, 200)
    engine = create_async_engine(
        test_db_url, connect_args={"options": "-c enable_seqscan=off -c enable_sort=off"}
    )
    try:
        async with engine.connect() as conn:
            result = await vector_search(
                conn, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS, limit=5
            )
            assert _ids(result) == {active_chunk}

        async with engine.connect() as conn:
            await conn.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
            plan = (
                await conn.execute(
                    text("EXPLAIN " + repository.VECTOR_SEARCH_SQL),
                    {
                        "allowed_tiers": [t.value for t in BOTH_TIERS],
                        "query_embedding": vec_literal(QUERY_VEC),
                        "vector_limit": 5,
                    },
                )
            ).scalars().all()
        assert any("chunks_embedding_hnsw" in line for line in plan), "\n".join(plan)
    finally:
        await engine.dispose()


# --- tier scope: same FROM/WHERE as the version filter --------------------------------------


def _seed_general_and_internal(engine: Engine) -> tuple[uuid.UUID, uuid.UUID]:
    with engine.begin() as conn:
        pub = make_document(conn, "public-doc", tier="general")
        pub_v = make_version(conn, pub, 1, "active")
        point_active(conn, pub, pub_v)
        general_chunk = make_chunk(
            conn, pub_v, 0, "refund policy public", unit_vector(0, blend=0.9)  # farther
        )
        sec = make_document(conn, "internal-doc", tier="internal")
        sec_v = make_version(conn, sec, 1, "active")
        point_active(conn, sec, sec_v)
        # Exact vector match and more query-term hits: outranks the general chunk on both branches.
        internal_chunk = make_chunk(
            conn, sec_v, 0, "refund policy refund policy internal margins", unit_vector(0)
        )
    return general_chunk, internal_chunk


async def test_general_tier_never_retrieves_internal_even_with_a_better_score(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    general_chunk, internal_chunk = _seed_general_and_internal(db_engine)
    async with async_engine.connect() as conn:
        vec = await vector_search(
            conn, query_embedding=QUERY_VEC, allowed_tiers=[Tier.GENERAL], limit=10
        )
        fts = await fulltext_search(
            conn, query_text=QUERY_TEXT, allowed_tiers=[Tier.GENERAL], limit=10
        )
    hybrid = await hybrid_search(
        async_engine, query_text=QUERY_TEXT, query_embedding=QUERY_VEC, allowed_tiers=[Tier.GENERAL]
    )
    for result in (vec, fts, hybrid):
        assert _ids(result) == {general_chunk}
        assert internal_chunk not in _ids(result)


async def test_internal_tier_sees_both_and_ranks_the_better_match_first(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    general_chunk, internal_chunk = _seed_general_and_internal(db_engine)
    async with async_engine.connect() as conn:
        vec = await vector_search(
            conn, query_embedding=QUERY_VEC, allowed_tiers=BOTH_TIERS, limit=10
        )
    assert [e.chunk_id for e in vec] == [internal_chunk, general_chunk]  # by ascending distance


async def test_empty_allowed_tiers_fails_closed(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    _seed_four_states(db_engine)
    async with async_engine.connect() as conn:
        assert (
            await vector_search(conn, query_embedding=QUERY_VEC, allowed_tiers=[], limit=10) == []
        )
        assert await fulltext_search(conn, query_text=QUERY_TEXT, allowed_tiers=[], limit=10) == []
    assert (
        await hybrid_search(
            async_engine, query_text=QUERY_TEXT, query_embedding=QUERY_VEC, allowed_tiers=[]
        )
        == []
    )


# --- full-text semantics (decision D) ---------------------------------------------------------


async def test_fulltext_matches_on_any_query_term_not_all(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    """Natural-language questions must not need every word present (AND would return nothing)."""
    seeded = _seed_four_states(db_engine)
    async with async_engine.connect() as conn:
        result = await fulltext_search(
            conn, query_text="refund banana zebra", allowed_tiers=BOTH_TIERS, limit=10
        )
    assert _ids(result) == {seeded.active_chunk}


async def test_fulltext_ranks_more_matching_terms_higher(
    db_engine: Engine, async_engine: AsyncEngine
) -> None:
    with db_engine.begin() as conn:
        doc = make_document(conn, "rank")
        v = make_version(conn, doc, 1, "active")
        point_active(conn, doc, v)
        one = make_chunk(conn, v, 0, "refund", unit_vector(2))
        two = make_chunk(conn, v, 1, "refund policy premium", unit_vector(3))
    async with async_engine.connect() as conn:
        result = await fulltext_search(
            conn, query_text="refund policy premium", allowed_tiers=BOTH_TIERS, limit=10
        )
    assert [e.chunk_id for e in result] == [two, one]


@pytest.mark.parametrize(
    "hostile", ["refund & ! ( | :* <->", "'; DROP TABLE chunks; --", "((((", "!!!", "   ", ""]
)
async def test_fulltext_tolerates_tsquery_syntax_in_user_text(
    db_engine: Engine, async_engine: AsyncEngine, hostile: str
) -> None:
    _seed_four_states(db_engine)
    async with async_engine.connect() as conn:
        result = await fulltext_search(
            conn, query_text=hostile, allowed_tiers=BOTH_TIERS, limit=10
        )
        assert isinstance(result, list)
        # And nothing was executed as SQL:
        assert (await conn.execute(text("SELECT count(*) FROM chunks"))).scalar_one() == 4
