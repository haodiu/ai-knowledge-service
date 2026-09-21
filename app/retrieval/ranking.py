"""Reciprocal Rank Fusion of the vector and full-text rankings (Plan §9.1 step 6). Pure."""
from collections.abc import Sequence
from dataclasses import replace

from app.retrieval.schemas import Evidence

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Evidence]], *, limit: int, k: int = RRF_K
) -> list[Evidence]:
    """Fuse ranked lists by chunk_id; score = sum over lists of 1 / (k + rank), rank from 1.

    Ties are broken by chunk_id so the output is deterministic. The returned `score` is the fused
    RRF score, not either branch's raw score.
    """
    scores: dict[object, float] = {}
    first_seen: dict[object, Evidence] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(item.chunk_id, item)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], str(cid)))
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered[:limit]]
