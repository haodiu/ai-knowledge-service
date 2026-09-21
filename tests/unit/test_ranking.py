import uuid
from dataclasses import replace

import pytest

from app.retrieval.ranking import RRF_K, reciprocal_rank_fusion
from app.retrieval.schemas import Evidence


def _ev(n: int) -> Evidence:
    return Evidence(
        chunk_id=uuid.UUID(int=n),
        document_version_id=uuid.UUID(int=1000 + n),
        document_id=uuid.UUID(int=2000),
        title="t",
        version_no=1,
        chunk_index=n,
        text=f"chunk {n}",
        score=0.0,
    )


def test_chunk_in_both_rankings_beats_chunks_in_one() -> None:
    a, b, c = _ev(1), _ev(2), _ev(3)
    fused = reciprocal_rank_fusion([[a, b], [c, b]], limit=10)
    assert [e.chunk_id for e in fused][0] == b.chunk_id


def test_score_is_sum_of_reciprocal_ranks() -> None:
    a = _ev(1)
    (out,) = reciprocal_rank_fusion([[a], [_ev(9), a]], limit=1)
    assert out.score == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))


def test_limit_and_deterministic_tie_break() -> None:
    items = [_ev(3), _ev(1), _ev(2)]
    once = reciprocal_rank_fusion([items], limit=2)
    assert [e.chunk_id for e in once] == [items[0].chunk_id, items[1].chunk_id]  # rank order
    tie = reciprocal_rank_fusion([[_ev(2)], [_ev(1)]], limit=10)
    assert [e.chunk_id.int for e in tie] == [1, 2]  # equal scores -> by chunk_id


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], limit=5) == []
    assert reciprocal_rank_fusion([[], []], limit=5) == []


def test_does_not_mutate_inputs() -> None:
    a = _ev(1)
    before = replace(a)
    reciprocal_rank_fusion([[a]], limit=1)
    assert a == before
