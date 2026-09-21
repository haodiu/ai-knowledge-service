import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedColumn, mapped_column

# Deterministic constraint names keep Alembic autogenerate and downgrades stable.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Must equal the VECTOR(n) width of chunks.embedding (Plan §5.1). index_config_hash covers the
# embedding dimensions, so changing this is a migration plus a full re-index, never a tweak.
EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _uuid_pk() -> MappedColumn[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


_NOW = text("now()")
# Module-level: inside Chunk the `text` column shadows sqlalchemy.text.
_EMPTY_JSON = text("'{}'::jsonb")


class Document(Base):
    """Stable document identity. Content lives in DocumentVersion (Plan §5.1)."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("tier IN ('general', 'internal')", name="tier"),
        CheckConstraint("status IN ('active', 'archived')", name="status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    external_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    # DEFERRABLE INITIALLY DEFERRED: documents <-> document_versions reference each other, and
    # activation rewrites both in one transaction. Checked at COMMIT only (Plan §5.1).
    active_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_versions.id",
            name="documents_active_version_fk",
            use_alter=True,  # cycle with document_versions.document_id
            deferrable=True,
            initially="DEFERRED",
        ),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        CheckConstraint("status IN ('building', 'active', 'superseded', 'failed')", name="status"),
        # Explicit names: the naming convention would give both the same name (column_0 only).
        UniqueConstraint(
            "document_id", "version_no", name="uq_document_versions_document_id_version_no"
        ),
        UniqueConstraint(
            "document_id",
            "content_hash",
            "index_config_hash",
            name="uq_document_versions_document_id_content_hash_index_config_hash",
        ),
        Index(
            "document_versions_one_active_per_document",
            "document_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        Index("document_versions_document_status_idx", "document_id", "status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    chunking_version: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    index_config_hash: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id", "chunk_index", name="uq_chunks_document_version_id_chunk_index"
        ),
        Index("chunks_document_version_idx", "document_version_id"),
        Index(
            "chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("chunks_text_search_gin", "text_search", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Generated so ingestion code cannot forget to compute it (Plan §5.1).
    text_search: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple'::regconfig, coalesce(text, ''))", persisted=True),
    )
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSON
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class KnowledgeBaseState(Base):
    """Singleton row; knowledge_version is bumped inside the activation transaction (Plan §5.5)."""

    __tablename__ = "knowledge_base_state"
    __table_args__ = (CheckConstraint("singleton_id = 1", name="singleton"),)

    singleton_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, server_default="1")
    knowledge_version: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
