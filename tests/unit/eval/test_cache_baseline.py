"""Pure computation/formatting in app.eval.cache_baseline -- no DB (see
tests/integration/test_cache_baseline.py for the SQL itself)."""
from app.eval.cache_baseline import (
    CAVEAT,
    CacheBaselineReport,
    DuplicateQuestionStats,
    EmbeddingLatencyStats,
    RemainderLatencyStats,
    RoleLatencyStats,
    _mean,
    _p95,
    render_markdown,
)


def test_p95_of_empty_is_none() -> None:
    assert _p95([]) is None


def test_p95_nearest_rank_on_a_small_sorted_set() -> None:
    # nearest-rank p95 of 1..100, 0-indexed: round(0.95 * 99) = 94 -> value 95
    assert _p95(list(range(1, 101))) == 95.0


def test_p95_single_value() -> None:
    assert _p95([42]) == 42.0


def test_mean_of_empty_is_none() -> None:
    assert _mean([]) is None


def test_mean_of_values() -> None:
    assert _mean([1, 2, 3]) == 2.0


def _report(**over: object) -> CacheBaselineReport:
    defaults: dict[str, object] = dict(
        duplicate_questions=DuplicateQuestionStats(0, 0, None, []),
        embedding=EmbeddingLatencyStats(0, None, None, None),
        role_latency=[],
        remainder=RemainderLatencyStats(None, None),
        mean_turn_latency_ms=None,
    )
    defaults.update(over)
    return CacheBaselineReport(**defaults)  # type: ignore[arg-type]


def test_render_markdown_always_includes_the_non_production_caveat() -> None:
    assert CAVEAT in render_markdown(_report())


def test_render_markdown_reports_no_data_yet_instead_of_crashing_on_empty_input() -> None:
    out = render_markdown(_report())
    assert "n/a (no data yet)" in out


def test_render_markdown_surfaces_duplicate_ratio_and_top_repeated_questions() -> None:
    out = render_markdown(
        _report(
            duplicate_questions=DuplicateQuestionStats(
                total_turns=10, duplicated_turns=4, duplicate_ratio=0.4,
                top_repeated=[("nghỉ phép bao nhiêu ngày?", 3)],
            )
        )
    )
    assert "0.4" in out
    assert "nghỉ phép bao nhiêu ngày?" in out
    assert "3×" in out


def test_render_markdown_surfaces_per_role_latency() -> None:
    out = render_markdown(
        _report(role_latency=[RoleLatencyStats("answer", count=5, mean_ms=3000.0, p95_ms=6000.0)])
    )
    assert "answer" in out and "3e+03ms" in out


def test_render_markdown_never_hides_that_postgresql_load_is_not_computed() -> None:
    out = render_markdown(_report())
    assert "pg_stat_statements" in out
