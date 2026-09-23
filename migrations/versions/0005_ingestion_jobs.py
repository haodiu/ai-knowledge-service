"""ingestion_jobs: queue tracking for Celery ingestion (Plan §5.3, §12.6 rule 7)

`document_version_id` is UNIQUE and ON DELETE SET NULL: one job per version attempt, and the
retention cleanup of Plan §12.7 must be free to delete an old version without touching this row
(audit fields and idempotency_key survive; only the pointer is cleared). `document_id` has a plain
(non-cascading) FK: a job stays queryable even after its document is archived.

`idempotency_key = document_id + version_no + content_hash + index_config_hash` (Plan §12.6) is the
job-creation-level idempotency backstop, on top of `document_versions`' own
`UNIQUE(document_id, content_hash, index_config_hash)`.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-23
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True)),
        sa.Column("version_no", sa.Integer, nullable=False),
        sa.Column("content_hash", sa.Text, nullable=False),
        sa.Column("idempotency_key", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("celery_task_id", sa.Text),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("error_code", sa.Text),
        sa.Column("error_message", sa.Text),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_jobs")),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name=op.f("fk_ingestion_jobs_document_id_documents")
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name=op.f("fk_ingestion_jobs_document_version_id_document_versions"),
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint(
            "document_version_id", name=op.f("uq_ingestion_jobs_document_version_id")
        ),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_ingestion_jobs_idempotency_key")),
        sa.CheckConstraint(
            "status IN ('queued', 'processing', 'retrying', 'completed', 'failed', 'superseded')",
            name=op.f("ck_ingestion_jobs_status"),
        ),
    )
    op.create_index(
        "ingestion_jobs_status_available_at_idx", "ingestion_jobs", ["status", "available_at"]
    )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
