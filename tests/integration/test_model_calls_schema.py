"""conversations / turns / model_calls exactly as Plan §5.4 (turn_sources: test_turn_sources.py).

model_calls is operational metadata only: the table must have nowhere to put a raw prompt or a
raw response (Plan §5.4, §17, DoD).
"""
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.integration

MODEL_CALL_COLUMNS = {
    "id", "turn_id", "purpose", "provider", "model_name", "prompt_version",
    "input_tokens", "output_tokens", "latency_ms", "status", "error_code", "created_at",
}


def _turn(conn) -> str:  # type: ignore[no-untyped-def]
    conv = conn.execute(
        text("INSERT INTO conversations (user_id) VALUES ('u1') RETURNING id")
    ).scalar_one()
    return conn.execute(
        text("INSERT INTO turns (conversation_id, question, graph_status) "
             "VALUES (:c, 'q', 'running') RETURNING id"),
        {"c": conv},
    ).scalar_one()


def _call(conn, turn_id, **over):  # type: ignore[no-untyped-def]
    row = {"t": turn_id, "purpose": "plan", "status": "ok", **over}
    conn.execute(
        text("INSERT INTO model_calls (turn_id, purpose, provider, model_name, prompt_version, "
             "latency_ms, status) VALUES (:t, :purpose, 'p', 'm', 'v1', 5, :status)"),
        row,
    )


def test_model_calls_columns_are_metadata_only(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        cols = {
            r[0]: r[1]
            for r in conn.execute(
                text("SELECT column_name, data_type FROM information_schema.columns "
                     "WHERE table_name = 'model_calls'")
            )
        }
    assert set(cols) == MODEL_CALL_COLUMNS
    assert "jsonb" not in cols.values() and "json" not in cols.values()  # no payload blob
    # the only free-text columns are identifiers, never prompt/response content
    text_cols = {c for c, t in cols.items() if t == "text"}
    assert text_cols == {"purpose", "provider", "model_name", "prompt_version", "status",
                         "error_code"}


def test_purpose_and_status_are_constrained(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        turn = _turn(conn)
        for purpose in ("plan", "grade", "answer", "repair"):
            _call(conn, turn, purpose=purpose)
    for bad in ({"purpose": "embed"}, {"status": "weird"}):
        with pytest.raises(IntegrityError), db_engine.begin() as conn:
            _call(conn, _turn(conn), **bad)


def test_deleting_a_turn_cascades_to_its_model_calls(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        turn = _turn(conn)
        _call(conn, turn)
        conn.execute(text("DELETE FROM turns WHERE id = :t"), {"t": turn})
        assert conn.execute(text("SELECT count(*) FROM model_calls")).scalar_one() == 0


def test_model_calls_requires_an_existing_turn(db_engine: Engine) -> None:
    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        _call(conn, "00000000-0000-0000-0000-000000000000")

