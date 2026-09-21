"""turn_sources: immutable citation snapshots (Plan §5.4)

document_id / document_version_id / chunk_id have NO foreign keys, on purpose: they are a
snapshot, and the version cleanup of Plan §12.7 deletes the rows they used to point at. A FK
(especially with ON DELETE CASCADE) would silently delete citation history. Do not add one.
The only FK is turn_id -> turns.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-21
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "turn_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True)),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True)),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True)),
        sa.Column("document_title", sa.Text, nullable=False),
        sa.Column("version_no", sa.Integer, nullable=False),
        sa.Column("text_snapshot", sa.Text, nullable=False),
        sa.Column("metadata_snapshot", postgresql.JSONB, nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_turn_sources")),
        sa.ForeignKeyConstraint(
            ["turn_id"], ["turns.id"],
            name=op.f("fk_turn_sources_turn_id_turns"), ondelete="CASCADE",
        ),
        sa.UniqueConstraint("source_id", name=op.f("uq_turn_sources_source_id")),
        sa.UniqueConstraint(
            "turn_id", "document_version_id", "chunk_id",
            name=op.f("uq_turn_sources_turn_id_document_version_id_chunk_id"),
        ),
    )


def downgrade() -> None:
    op.drop_table("turn_sources")
