"""embedding_calls: operational metadata for embedding calls, online + offline (Plan §17, §18
Week 8)

Structural twin of `model_calls` (0003) / `tool_calls` (0006), with one difference: embeddings
happen on TWO independent paths with two different owners -- online query embeddings (`turn_id`)
and offline document embeddings during ingestion (`ingestion_job_id`) -- so exactly one of the two
FK columns is set, enforced by a CHECK constraint rather than two separate tables (which would
duplicate the schema for no benefit). No `estimated_cost` column, deliberately: matches
`model_calls`' own precedent (tokens only, no cost) so pricing has exactly one home, the Week 8
eval runner's pricing table, instead of drifting between two.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "embedding_calls",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True)),
        sa.Column("ingestion_job_id", postgresql.UUID(as_uuid=True)),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("provider", sa.Text, nullable=False),
        sa.Column("model_version", sa.Text, nullable=False),
        sa.Column("batch_size", sa.Integer, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("error_code", sa.Text),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_embedding_calls")),
        sa.ForeignKeyConstraint(
            ["turn_id"],
            ["turns.id"],
            name=op.f("fk_embedding_calls_turn_id_turns"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["ingestion_job_id"],
            ["ingestion_jobs.id"],
            name=op.f("fk_embedding_calls_ingestion_job_id_ingestion_jobs"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("kind IN ('document', 'query')", name=op.f("ck_embedding_calls_kind")),
        sa.CheckConstraint("status IN ('ok', 'error')", name=op.f("ck_embedding_calls_status")),
        sa.CheckConstraint(
            "(turn_id IS NULL) <> (ingestion_job_id IS NULL)",
            name=op.f("ck_embedding_calls_exactly_one_owner"),
        ),
    )


def downgrade() -> None:
    op.drop_table("embedding_calls")
