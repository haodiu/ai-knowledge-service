"""Schema-level guarantees from Plan §5.1/§5.5 that the rest of the design leans on."""
import uuid

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from tests.integration.conftest import (
    make_chunk,
    make_document,
    make_version,
    point_active,
    unit_vector,
    vec_literal,
)

pytestmark = pytest.mark.integration


def test_active_version_fk_is_deferrable_initially_deferred(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT condeferrable, condeferred FROM pg_constraint "
                "WHERE conname = 'documents_active_version_fk'"
            )
        ).one()
    assert (row.condeferrable, row.condeferred) == (True, True)


def test_active_version_fk_is_only_checked_at_commit(db_engine: Engine) -> None:
    """Point a document at a version row that is inserted *after* the pointer (same txn)."""
    version_id = uuid.uuid4()
    with db_engine.begin() as conn:
        doc_id = conn.execute(
            text(
                "INSERT INTO documents (external_id, title, tier, active_version_id) "
                "VALUES ('cycle', 't', 'general', :v) RETURNING id"
            ),
            {"v": version_id},
        ).scalar_one()
        conn.execute(
            text(
                "INSERT INTO document_versions (id, document_id, version_no, content, "
                "content_hash, chunking_version, embedding_model, index_config_hash, status) "
                "VALUES (:v, :d, 1, 'c', 'h', 'c', 'm', 'i', 'active')"
            ),
            {"v": version_id, "d": doc_id},
        )
    with db_engine.connect() as conn:
        stored = conn.execute(
            text("SELECT active_version_id FROM documents WHERE id = :d"), {"d": doc_id}
        ).scalar_one()
    assert stored == version_id


def test_dangling_active_version_id_is_rejected_at_commit(db_engine: Engine) -> None:
    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO documents (external_id, title, tier, active_version_id) "
                "VALUES ('dangling', 't', 'general', gen_random_uuid())"
            )
        )
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM documents")).scalar_one() == 0


def test_at_most_one_active_version_per_document(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        doc_id = make_document(conn, "one-active")
        make_version(conn, doc_id, 1, "active")
    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        make_version(conn, doc_id, 2, "active")


def test_text_search_is_generated_and_populated(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        doc_id = make_document(conn, "fts")
        version_id = make_version(conn, doc_id, 1, "building")
        chunk_id = make_chunk(conn, version_id, 0, "Refund policy for Premium", unit_vector(0))
    with db_engine.connect() as conn:
        tsv = conn.execute(
            text("SELECT text_search::text FROM chunks WHERE id = :c"), {"c": chunk_id}
        ).scalar_one()
        assert tsv is not None
        assert "'refund'" in tsv and "'premium'" in tsv  # 'simple' config: lowercased, unstemmed
        hit = conn.execute(
            text(
                "SELECT count(*) FROM chunks "
                "WHERE text_search @@ plainto_tsquery('simple', 'refund')"
            )
        ).scalar_one()
        assert hit == 1


def test_text_search_cannot_be_written_directly(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        doc_id = make_document(conn, "fts-ro")
        version_id = make_version(conn, doc_id, 1, "building")
    with pytest.raises(DBAPIError), db_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO chunks "
                "(document_version_id, chunk_index, text, text_search, embedding) "
                "VALUES (:v, 0, 'x', to_tsvector('simple', 'y'), CAST(:e AS vector))"
            ),
            {"v": version_id, "e": vec_literal(unit_vector(0))},
        )


def test_duplicate_chunk_index_within_a_version_is_rejected(db_engine: Engine) -> None:
    """DB-level backstop for at-least-once redelivery (invariant 10)."""
    with db_engine.begin() as conn:
        doc_id = make_document(conn, "dup")
        version_id = make_version(conn, doc_id, 1, "building")
        make_chunk(conn, version_id, 0, "a", unit_vector(0))
    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        make_chunk(conn, version_id, 0, "b", unit_vector(1))


def test_embedding_dimension_is_enforced(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        doc_id = make_document(conn, "dim")
        version_id = make_version(conn, doc_id, 1, "building")
    with pytest.raises(DBAPIError), db_engine.begin() as conn:
        make_chunk(conn, version_id, 0, "short", [1.0, 0.0, 0.0])


def test_knowledge_base_state_is_a_seeded_singleton(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM knowledge_base_state")).scalar_one() == 1
        kv = conn.execute(text("SELECT knowledge_version FROM knowledge_base_state")).scalar_one()
        assert kv == 0
    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        conn.execute(text("INSERT INTO knowledge_base_state (singleton_id) VALUES (2)"))


def test_helper_can_build_a_document_with_active_pointer(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        doc_id = make_document(conn, "helper")
        version_id = make_version(conn, doc_id, 1, "active")
        point_active(conn, doc_id, version_id)
    with db_engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT active_version_id FROM documents WHERE id = :d"), {"d": doc_id}
            ).scalar_one()
            == version_id
        )
