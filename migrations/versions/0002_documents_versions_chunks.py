"""documents, document_versions, chunks, knowledge_base_state (Plan §5.1, §5.5)

`ingestion_jobs` is deliberately absent: its columns are queue concepts (Week 5, Plan §18).
Types and constraints follow Plan §5.1 verbatim; the only additions are explicit constraint
names for the two multi-column UNIQUEs (the naming convention would collide on them).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-21
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1536  # frozen here on purpose: a migration must not follow later edits to models


def _uuid_pk(name: str) -> sa.Column[sa.UUID]:
    return sa.Column(
        name,
        postgresql.UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    )


def _ts(name: str, *, nullable: bool = False, default_now: bool = True) -> sa.Column[sa.DateTime]:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        nullable=nullable,
        server_default=sa.text("now()") if default_now else None,
    )


def upgrade() -> None:
    # No FK on active_version_id yet: documents and document_versions reference each other.
    op.create_table(
        "documents",
        _uuid_pk("id"),
        sa.Column("external_id", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("tier", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="active"),
        sa.Column("active_version_id", postgresql.UUID(as_uuid=True)),
        _ts("created_at"),
        _ts("updated_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("external_id", name=op.f("uq_documents_external_id")),
        sa.CheckConstraint("tier IN ('general', 'internal')", name=op.f("ck_documents_tier")),
        sa.CheckConstraint("status IN ('active', 'archived')", name=op.f("ck_documents_status")),
    )

    op.create_table(
        "document_versions",
        _uuid_pk("id"),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_no", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_hash", sa.Text, nullable=False),
        sa.Column("chunking_version", sa.Text, nullable=False),
        sa.Column("embedding_model", sa.Text, nullable=False),
        sa.Column("index_config_hash", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        _ts("created_at"),
        _ts("activated_at", nullable=True, default_now=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_versions")),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_versions_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "document_id", "version_no", name="uq_document_versions_document_id_version_no"
        ),
        sa.UniqueConstraint(
            "document_id",
            "content_hash",
            "index_config_hash",
            name="uq_document_versions_document_id_content_hash_index_config_hash",
        ),
        sa.CheckConstraint(
            "status IN ('building', 'active', 'superseded', 'failed')",
            name=op.f("ck_document_versions_status"),
        ),
    )

    op.create_foreign_key(
        "documents_active_version_fk",
        "documents",
        "document_versions",
        ["active_version_id"],
        ["id"],
        deferrable=True,
        initially="DEFERRED",
    )

    op.create_index(
        "document_versions_one_active_per_document",
        "document_versions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "document_versions_document_status_idx", "document_versions", ["document_id", "status"]
    )

    op.create_table(
        "chunks",
        _uuid_pk("id"),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column(
            "text_search",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('simple'::regconfig, coalesce(text, ''))", persisted=True),
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "metadata_json",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        _ts("created_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name=op.f("fk_chunks_document_version_id_document_versions"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "document_version_id", "chunk_index", name="uq_chunks_document_version_id_chunk_index"
        ),
    )
    op.create_index("chunks_document_version_idx", "chunks", ["document_version_id"])
    op.create_index(
        "chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("chunks_text_search_gin", "chunks", ["text_search"], postgresql_using="gin")

    op.create_table(
        "knowledge_base_state",
        sa.Column("singleton_id", sa.SmallInteger, nullable=False, server_default="1"),
        sa.Column("knowledge_version", sa.BigInteger, nullable=False, server_default="0"),
        _ts("updated_at"),
        sa.PrimaryKeyConstraint("singleton_id", name=op.f("pk_knowledge_base_state")),
        sa.CheckConstraint("singleton_id = 1", name=op.f("ck_knowledge_base_state_singleton")),
    )
    # Activation does UPDATE ... knowledge_version + 1; without this row it would touch 0 rows.
    op.execute("INSERT INTO knowledge_base_state (singleton_id) VALUES (1)")


def downgrade() -> None:
    op.drop_table("knowledge_base_state")
    op.drop_table("chunks")
    # Break the documents <-> document_versions cycle before dropping either table.
    op.drop_constraint("documents_active_version_fk", "documents", type_="foreignkey")
    op.drop_table("document_versions")
    op.drop_table("documents")
