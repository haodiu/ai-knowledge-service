"""build_evidence: dedupe, rank, cap, and never cut a chunk in half (Plan §11.5)."""
from dataclasses import replace

from app.graph.limits import MAX_RETRIEVED_CHUNKS
from app.graph.nodes.evidence import build_evidence
from tests.unit.ai.helpers import make_evidence


def test_duplicates_are_removed_by_the_version_chunk_pair() -> None:
    ev = make_evidence(2)
    out = build_evidence([], [ev[0], ev[0], ev[1]])
    assert len(out) == 2


def test_same_chunk_id_under_another_version_is_a_different_source() -> None:
    e = make_evidence(1)[0]
    other = replace(e, document_version_id=make_evidence(1)[0].document_version_id)
    assert len(build_evidence([], [e, other])) == 2


def test_results_are_ordered_by_score_and_capped_at_the_plan_limit() -> None:
    ev = [replace(e, score=float(i)) for i, e in enumerate(make_evidence(12))]
    out = build_evidence([], ev)
    assert len(out) == MAX_RETRIEVED_CHUNKS == 8
    assert [e.score for e in out] == sorted((e.score for e in out), reverse=True)
    assert out[0].score == 11.0


def test_second_attempt_is_a_union_with_the_first() -> None:
    first, second = make_evidence(2), make_evidence(2)
    out = build_evidence(first, second + [first[0]])
    assert {(e.document_version_id, e.chunk_id) for e in out} == {
        (e.document_version_id, e.chunk_id) for e in first + second}


def test_char_budget_drops_whole_trailing_chunks_and_never_truncates_text() -> None:
    ev = [replace(e, text="x" * 1000, score=10.0 - i) for i, e in enumerate(make_evidence(5))]
    out = build_evidence([], ev, max_chars=2500)
    assert len(out) == 2
    assert all(len(e.text) == 1000 for e in out)  # kept intact, not cut


def test_the_top_chunk_is_kept_even_if_it_alone_exceeds_the_budget() -> None:
    (e,) = make_evidence(1)
    big = replace(e, text="y" * 5000)
    assert build_evidence([], [big], max_chars=100) == [big]


def test_empty_in_empty_out() -> None:
    assert build_evidence([], []) == []


def test_a_rewrite_that_scores_lower_still_displaces_older_evidence_at_the_cap() -> None:
    """RRF scores from different queries are not comparable: the rewrite's finds must survive."""
    old = [replace(e, score=0.9) for e in make_evidence(8)]
    found = [replace(e, score=0.1) for e in make_evidence(2)]
    out = build_evidence(old, found)
    assert len(out) == 8
    assert all(f in out for f in found)  # both new chunks kept despite the lower score
    assert sum(1 for o in old if o in out) == 6  # the cap trimmed the older evidence instead
