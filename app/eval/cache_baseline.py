"""Cache baseline report (Plan §13.2/§22, CLAUDE.md invariant #12).

MVP does not cache embeddings/retrieval/answers. Before adding any such cache, the Plan requires
measuring, from real traffic: (1) the duplicate-normalized-question ratio, (2) embedding
latency/cost as a share of total turn latency, (3) retrieval p95 / PostgreSQL load, and (4) the
expected improvement vs. invalidation complexity. This module computes what's answerable from data
already recorded in `turns`/`model_calls`/`embedding_calls` (no new instrumentation, no migration).
PostgreSQL load itself isn't derivable from these tables -- the report says so rather than
fabricating a number; `pg_stat_statements` is the tool for that if it's ever needed.

    python -m app.eval.cache_baseline

Read-only: this script never writes anything. It answers "does a cache look justified yet",
nothing more -- it does not implement one (that's a separate, reviewed decision; see invariant #12
and the "don't add caching while you're in there" working agreement in CLAUDE.md).
"""
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Engine, text

from app.db.session import create_sync_engine
from app.settings import get_settings

CAVEAT = (
    "⚠️  These numbers reflect whatever traffic is in this database right now -- manual test "
    "questions during development, not production usage. This is a diagnostic/demo run, NOT a "
    "production baseline. Re-run this against real traffic before deciding whether a cache is "
    "justified (Plan §13.2/§22)."
)


def _p95(values: Sequence[int]) -> float | None:
    """Same simple nearest-rank p95 as `app.eval.report._p95` -- duplicated rather than imported
    across modules since that helper is private to `report.py`."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return float(ordered[index])


def _mean(values: Sequence[int]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class DuplicateQuestionStats:
    total_turns: int
    duplicated_turns: int  # turns whose normalized question appears more than once
    duplicate_ratio: float | None
    top_repeated: list[tuple[str, int]]  # (normalized question, count), most-repeated first


@dataclass(frozen=True)
class RoleLatencyStats:
    purpose: str
    count: int
    mean_ms: float | None
    p95_ms: float | None


@dataclass(frozen=True)
class EmbeddingLatencyStats:
    count: int
    mean_ms: float | None
    p95_ms: float | None
    share_of_mean_turn_latency: float | None  # mean_ms / mean turn latency_ms


@dataclass(frozen=True)
class RemainderLatencyStats:
    """turns.latency_ms minus the sum of that turn's model_calls + embedding_calls latency --
    an approximation of retrieval SQL + graph/asyncio/DB overhead, not a pure retrieval number."""

    mean_ms: float | None
    p95_ms: float | None


@dataclass(frozen=True)
class CacheBaselineReport:
    duplicate_questions: DuplicateQuestionStats
    embedding: EmbeddingLatencyStats
    role_latency: list[RoleLatencyStats]
    remainder: RemainderLatencyStats
    mean_turn_latency_ms: float | None


def _duplicate_question_stats(engine: Engine, *, top_n: int = 10) -> DuplicateQuestionStats:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT lower(trim(question)) AS q, count(*) AS n "
                "FROM turns GROUP BY lower(trim(question))"
            )
        ).all()
    total_turns = sum(r.n for r in rows)
    duplicated_turns = sum(r.n for r in rows if r.n > 1)
    ratio = duplicated_turns / total_turns if total_turns else None
    top_repeated = sorted(((r.q, r.n) for r in rows if r.n > 1), key=lambda kv: -kv[1])[:top_n]
    return DuplicateQuestionStats(total_turns, duplicated_turns, ratio, top_repeated)


def _embedding_stats(engine: Engine, mean_turn_latency_ms: float | None) -> EmbeddingLatencyStats:
    with engine.connect() as conn:
        latencies = [
            r[0]
            for r in conn.execute(
                text(
                    "SELECT latency_ms FROM embedding_calls "
                    "WHERE kind = 'query' AND status = 'ok'"
                )
            ).all()
        ]
    mean_ms = _mean(latencies)
    share = mean_ms / mean_turn_latency_ms if mean_ms is not None and mean_turn_latency_ms else None
    return EmbeddingLatencyStats(len(latencies), mean_ms, _p95(latencies), share)


def _role_latency_stats(engine: Engine) -> list[RoleLatencyStats]:
    with engine.connect() as conn:
        purposes = [r[0] for r in conn.execute(text("SELECT DISTINCT purpose FROM model_calls"))]
        stats = []
        for purpose in sorted(purposes):
            latencies = [
                r[0]
                for r in conn.execute(
                    text(
                        "SELECT latency_ms FROM model_calls WHERE purpose = :p AND status = 'ok'"
                    ),
                    {"p": purpose},
                ).all()
            ]
            stats.append(
                RoleLatencyStats(purpose, len(latencies), _mean(latencies), _p95(latencies))
            )
    return stats


def _remainder_stats(engine: Engine) -> RemainderLatencyStats:
    """Per turn: turns.latency_ms - sum(model_calls.latency_ms) - sum(embedding_calls.latency_ms).
    Only over turns that actually finished with a recorded latency."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT t.latency_ms - COALESCE(m.total, 0) - COALESCE(e.total, 0) AS remainder "
                "FROM turns t "
                "LEFT JOIN (SELECT turn_id, sum(latency_ms) AS total FROM model_calls "
                "           GROUP BY turn_id) m ON m.turn_id = t.id "
                "LEFT JOIN (SELECT turn_id, sum(latency_ms) AS total FROM embedding_calls "
                "           WHERE turn_id IS NOT NULL GROUP BY turn_id) e ON e.turn_id = t.id "
                "WHERE t.latency_ms IS NOT NULL"
            )
        ).all()
    remainders = [r[0] for r in rows]
    return RemainderLatencyStats(_mean(remainders), _p95(remainders))


def _mean_turn_latency_ms(engine: Engine) -> float | None:
    with engine.connect() as conn:
        latencies = [
            r[0]
            for r in conn.execute(
                text("SELECT latency_ms FROM turns WHERE latency_ms IS NOT NULL")
            ).all()
        ]
    return _mean(latencies)


def compute_baseline(engine: Engine) -> CacheBaselineReport:
    mean_turn_latency_ms = _mean_turn_latency_ms(engine)
    return CacheBaselineReport(
        duplicate_questions=_duplicate_question_stats(engine),
        embedding=_embedding_stats(engine, mean_turn_latency_ms),
        role_latency=_role_latency_stats(engine),
        remainder=_remainder_stats(engine),
        mean_turn_latency_ms=mean_turn_latency_ms,
    )


def _fmt(value: float | None, suffix: str = "") -> str:
    return f"{value:.3g}{suffix}" if value is not None else "n/a (no data yet)"


def render_markdown(report: CacheBaselineReport) -> str:
    lines = [
        "# Cache baseline report",
        "",
        CAVEAT,
        "",
        "## 1. Duplicate-question ratio",
        f"- total turns: {report.duplicate_questions.total_turns}",
        f"- turns whose (normalized) question repeats: "
        f"{report.duplicate_questions.duplicated_turns}",
        f"- duplicate ratio: {_fmt(report.duplicate_questions.duplicate_ratio)}",
    ]
    if report.duplicate_questions.top_repeated:
        lines.append("- most-repeated questions:")
        lines += [f"  - ({n}×) {q!r}" for q, n in report.duplicate_questions.top_repeated]
    lines += [
        "",
        "## 2. Embedding latency/cost share",
        f"- query embedding calls: {report.embedding.count}",
        f"- mean latency: {_fmt(report.embedding.mean_ms, 'ms')}",
        f"- p95 latency: {_fmt(report.embedding.p95_ms, 'ms')}",
        f"- share of mean turn latency: {_fmt(report.embedding.share_of_mean_turn_latency)}",
        "",
        "## 3. Per-role LLM latency (plan/grade/answer/repair)",
    ]
    for r in report.role_latency:
        lines.append(
            f"- **{r.purpose}**: n={r.count}, mean={_fmt(r.mean_ms, 'ms')}, "
            f"p95={_fmt(r.p95_ms, 'ms')}"
        )
    lines += [
        "",
        "## 4. Remainder (retrieval SQL + graph/DB overhead, approximate)",
        f"- mean: {_fmt(report.remainder.mean_ms, 'ms')}",
        f"- p95: {_fmt(report.remainder.p95_ms, 'ms')}",
        "- this bucket is turns.latency_ms minus every model_calls/embedding_calls latency for "
        "that turn -- it is NOT a pure retrieval-SQL number (graph/asyncio/DB round-trip overhead "
        "is folded in too). Isolating retrieval SQL alone would need new instrumentation.",
        "",
        "## 5. PostgreSQL load",
        "- not computable from `turns`/`model_calls`/`embedding_calls` alone. If this number is "
        "ever needed, check `pg_stat_statements` (must be enabled on the server) rather than "
        "guessing from application tables.",
        "",
        f"Mean end-to-end turn latency: {_fmt(report.mean_turn_latency_ms, 'ms')}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    del argv  # no arguments yet; kept for CLI-entrypoint symmetry with the other app.*.cli scripts
    settings = get_settings()
    engine = create_sync_engine(settings)
    try:
        report = compute_baseline(engine)
    finally:
        engine.dispose()
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
