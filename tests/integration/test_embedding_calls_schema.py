"""embedding_calls exactly as migration 0007 / app/db/models.py::EmbeddingCall (Plan §17, §18
Week 8).

Structural twin of test_model_calls_schema.py/test_tool_calls_schema.py, plus the one thing that's
different here: embeddings happen on two independent paths (online query, offline document), so
exactly one of turn_id/ingestion_job_id is set -- never both, never neither.
"""
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from app.ai.embedding_recorder import EmbeddingCallRecord

pytestmark = pytest.mark.integration

EMBEDDING_CALL_COLUMNS = {
    "id", "turn_id", "ingestion_job_id", "kind", "provider", "model_version", "batch_size",
    "latency_ms", "status", "error_code", "created_at",
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


def _job(conn) -> str:  # type: ignore[no-untyped-def]
    doc = conn.execute(
        text("INSERT INTO documents (external_id, title, tier) "
             "VALUES ('docs/x.md', 'X', 'general') RETURNING id")
    ).scalar_one()
    return conn.execute(
        text("INSERT INTO ingestion_jobs (document_id, version_no, content_hash, "
             "idempotency_key, status) VALUES (:d, 1, 'h', 'k', 'queued') RETURNING id"),
        {"d": doc},
    ).scalar_one()


def _call(conn, *, turn_id=None, ingestion_job_id=None, **over):  # type: ignore[no-untyped-def]
    row = {
        "t": turn_id, "j": ingestion_job_id, "kind": "query", "provider": "fake",
        "mv": "fake-model", "batch": 1, "status": "ok", "err": None, **over,
    }
    conn.execute(
        text(
            "INSERT INTO embedding_calls (turn_id, ingestion_job_id, kind, provider, "
            "model_version, batch_size, latency_ms, status, error_code) "
            "VALUES (:t, :j, :kind, :provider, :mv, :batch, 5, :status, :err)"
        ),
        row,
    )


def test_embedding_calls_columns_are_metadata_only(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        cols = {
            r[0]: r[1]
            for r in conn.execute(
                text("SELECT column_name, data_type FROM information_schema.columns "
                     "WHERE table_name = 'embedding_calls'")
            )
        }
    assert set(cols) == EMBEDDING_CALL_COLUMNS
    assert "jsonb" not in cols.values() and "json" not in cols.values()  # no payload blob
    # no column for the query text or the resulting vector -- operational metadata only
    text_cols = {c for c, t in cols.items() if t == "text"}
    assert text_cols == {"kind", "provider", "model_version", "status", "error_code"}


def test_kind_and_status_are_constrained(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        turn = _turn(conn)
        for kind in ("document", "query"):
            _call(conn, turn_id=turn, kind=kind)
    for bad in ({"kind": "chunk"}, {"status": "weird"}):
        with pytest.raises(IntegrityError), db_engine.begin() as conn:
            _call(conn, turn_id=_turn(conn), **bad)


def test_exactly_one_owner_is_enforced(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        turn = _turn(conn)
        job = _job(conn)
        _call(conn, turn_id=turn)  # turn only -- fine
        _call(conn, ingestion_job_id=job, kind="document")  # job only -- fine

    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        _call(conn, turn_id=_turn(conn), ingestion_job_id=_job(conn))  # both -- rejected

    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        _call(conn)  # neither -- rejected


def test_deleting_a_turn_cascades_to_its_embedding_calls(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        turn = _turn(conn)
        _call(conn, turn_id=turn)
        conn.execute(text("DELETE FROM turns WHERE id = :t"), {"t": turn})
        assert conn.execute(text("SELECT count(*) FROM embedding_calls")).scalar_one() == 0


def test_deleting_an_ingestion_job_cascades_to_its_embedding_calls(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        job = _job(conn)
        _call(conn, ingestion_job_id=job, kind="document")
        conn.execute(text("DELETE FROM ingestion_jobs WHERE id = :j"), {"j": job})
        assert conn.execute(text("SELECT count(*) FROM embedding_calls")).scalar_one() == 0


async def test_insert_embedding_call_writes_exactly_the_recorded_fields(
    db_engine: Engine, async_engine: object
) -> None:
    from app.db import repositories

    with db_engine.begin() as conn:
        turn = _turn(conn)

    record = EmbeddingCallRecord(
        kind="query", provider="gemini", model_version="gemini-embedding-001",
        batch_size=1, latency_ms=42, status="error", error_code="timeout",
    )
    await repositories.insert_embedding_call(async_engine, turn_id=turn, record=record)  # type: ignore[arg-type]

    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT kind, provider, model_version, batch_size, latency_ms, status, "
                 "error_code FROM embedding_calls WHERE turn_id = :t"),
            {"t": turn},
        ).one()
    assert tuple(row) == ("query", "gemini", "gemini-embedding-001", 1, 42, "error", "timeout")
