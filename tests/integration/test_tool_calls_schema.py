"""tool_calls exactly as migration 0006 / app/db/models.py::ToolCall (Plan §10, §17, §18 Tuần 7).

Structural twin of test_model_calls_schema.py: tool_calls is operational metadata only -- the
table must have nowhere to put the raw subscription_id/customer_id value, only which kind of
identifier was used.
"""
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from app.tools.recorder import ToolCallRecord

pytestmark = pytest.mark.integration

TOOL_CALL_COLUMNS = {
    "id", "turn_id", "tool_name", "identifier_kind", "latency_ms", "status", "error_code",
    "created_at",
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
    row = {"t": turn_id, "name": "get_subscription", "kind": "customer_id", "status": "ok",
          "err": None, **over}
    conn.execute(
        text("INSERT INTO tool_calls (turn_id, tool_name, identifier_kind, latency_ms, status, "
             "error_code) VALUES (:t, :name, :kind, 5, :status, :err)"),
        row,
    )


def test_tool_calls_columns_are_metadata_only(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        cols = {
            r[0]: r[1]
            for r in conn.execute(
                text("SELECT column_name, data_type FROM information_schema.columns "
                     "WHERE table_name = 'tool_calls'")
            )
        }
    assert set(cols) == TOOL_CALL_COLUMNS
    assert "jsonb" not in cols.values() and "json" not in cols.values()  # no payload blob
    # only identifier_kind (which KIND was used), never the raw subscription_id/customer_id value
    text_cols = {c for c, t in cols.items() if t == "text"}
    assert text_cols == {"tool_name", "identifier_kind", "status", "error_code"}


def test_tool_name_and_identifier_kind_and_status_are_constrained(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        turn = _turn(conn)
        for kind in ("subscription_id", "customer_id"):
            _call(conn, turn, kind=kind)
    for bad in ({"name": "delete_subscription"}, {"kind": "email"}, {"status": "weird"}):
        with pytest.raises(IntegrityError), db_engine.begin() as conn:
            _call(conn, _turn(conn), **bad)


def test_deleting_a_turn_cascades_to_its_tool_calls(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        turn = _turn(conn)
        _call(conn, turn)
        conn.execute(text("DELETE FROM turns WHERE id = :t"), {"t": turn})
        assert conn.execute(text("SELECT count(*) FROM tool_calls")).scalar_one() == 0


def test_tool_calls_requires_an_existing_turn(db_engine: Engine) -> None:
    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        _call(conn, "00000000-0000-0000-0000-000000000000")


async def test_insert_tool_call_writes_exactly_the_recorded_fields(
    db_engine: Engine, async_engine: object
) -> None:
    from app.db import repositories

    with db_engine.begin() as conn:
        turn = _turn(conn)

    record = ToolCallRecord(
        tool_name="get_subscription", identifier_kind="subscription_id",
        latency_ms=42, status="error", error_code="tool_not_found",
    )
    await repositories.insert_tool_call(async_engine, turn, record)  # type: ignore[arg-type]

    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT tool_name, identifier_kind, latency_ms, status, error_code "
                 "FROM tool_calls WHERE turn_id = :t"),
            {"t": turn},
        ).one()
    assert tuple(row) == ("get_subscription", "subscription_id", 42, "error", "tool_not_found")
