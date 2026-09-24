"""app.eval.cache_baseline against a real, migrated schema (unit tests cover the pure calc/format
helpers in tests/unit/eval/test_cache_baseline.py)."""
import pytest
from sqlalchemy import Engine, text

from app.eval.cache_baseline import compute_baseline

pytestmark = pytest.mark.integration


def _turn(conn, question: str, latency_ms: int) -> str:  # type: ignore[no-untyped-def]
    conv = conn.execute(
        text("INSERT INTO conversations (user_id) VALUES ('u1') RETURNING id")
    ).scalar_one()
    return conn.execute(
        text(
            "INSERT INTO turns (conversation_id, question, graph_status, latency_ms) "
            "VALUES (:c, :q, 'answered', :l) RETURNING id"
        ),
        {"c": conv, "q": question, "l": latency_ms},
    ).scalar_one()


def _model_call(conn, turn_id, purpose: str, latency_ms: int) -> None:  # type: ignore[no-untyped-def]
    conn.execute(
        text(
            "INSERT INTO model_calls (turn_id, purpose, provider, model_name, prompt_version, "
            "latency_ms, status) VALUES (:t, :p, 'gemini', 'm', 'v1', :l, 'ok')"
        ),
        {"t": turn_id, "p": purpose, "l": latency_ms},
    )


def _embedding_call(conn, turn_id, latency_ms: int) -> None:  # type: ignore[no-untyped-def]
    conn.execute(
        text(
            "INSERT INTO embedding_calls (turn_id, kind, provider, model_version, batch_size, "
            "latency_ms, status) VALUES (:t, 'query', 'gemini', 'm', 1, :l, 'ok')"
        ),
        {"t": turn_id, "l": latency_ms},
    )


def test_no_data_yet_produces_a_report_with_no_crash(db_engine: Engine) -> None:
    report = compute_baseline(db_engine)
    assert report.duplicate_questions.total_turns == 0
    assert report.duplicate_questions.duplicate_ratio is None
    assert report.embedding.mean_ms is None
    assert report.role_latency == []


def test_duplicate_ratio_counts_normalized_repeats_case_and_whitespace_insensitively(
    db_engine: Engine,
) -> None:
    with db_engine.begin() as conn:
        _turn(conn, "Nghỉ phép bao nhiêu ngày?", 1000)
        _turn(conn, "  nghỉ phép bao nhiêu ngày?  ", 1000)  # same question, different case/spacing
        _turn(conn, "Chính sách khen thưởng là gì?", 1000)  # unique

    report = compute_baseline(db_engine)

    assert report.duplicate_questions.total_turns == 3
    assert report.duplicate_questions.duplicated_turns == 2
    assert report.duplicate_questions.duplicate_ratio == pytest.approx(2 / 3)
    assert report.duplicate_questions.top_repeated[0] == ("nghỉ phép bao nhiêu ngày?", 2)


def test_embedding_share_is_mean_embedding_over_mean_turn_latency(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        t1 = _turn(conn, "q1", 10_000)
        t2 = _turn(conn, "q2", 10_000)
        _embedding_call(conn, t1, 1_000)
        _embedding_call(conn, t2, 1_000)

    report = compute_baseline(db_engine)

    assert report.embedding.count == 2
    assert report.embedding.mean_ms == 1_000.0
    assert report.mean_turn_latency_ms == 10_000.0
    assert report.embedding.share_of_mean_turn_latency == pytest.approx(0.1)


def test_role_latency_is_grouped_by_purpose_and_only_counts_ok_calls(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        t = _turn(conn, "q1", 9_000)
        _model_call(conn, t, "plan", 2_000)
        _model_call(conn, t, "grade", 3_000)
        _model_call(conn, t, "answer", 4_000)
        conn.execute(
            text(
                "INSERT INTO model_calls (turn_id, purpose, provider, model_name, prompt_version, "
                "latency_ms, status, error_code) "
                "VALUES (:t, 'answer', 'gemini', 'm', 'v1', 999, 'error', 'unavailable')"
            ),
            {"t": t},
        )

    report = compute_baseline(db_engine)

    by_purpose = {r.purpose: r for r in report.role_latency}
    assert by_purpose["plan"].mean_ms == 2_000.0
    assert by_purpose["answer"].count == 1  # the error row is excluded
    assert by_purpose["answer"].mean_ms == 4_000.0


def test_remainder_subtracts_model_and_embedding_latency(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        t = _turn(conn, "q1", 10_000)
        _model_call(conn, t, "plan", 2_000)
        _model_call(conn, t, "grade", 2_000)
        _model_call(conn, t, "answer", 3_000)
        _embedding_call(conn, t, 500)

    report = compute_baseline(db_engine)

    assert report.remainder.mean_ms == pytest.approx(10_000 - 2_000 - 2_000 - 3_000 - 500)


def test_remainder_handles_a_turn_with_no_model_or_embedding_calls(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        _turn(conn, "q1", 5_000)

    report = compute_baseline(db_engine)

    assert report.remainder.mean_ms == 5_000.0
