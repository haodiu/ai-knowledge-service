"""conversations, turns, model_calls (Plan §5.4)

`turn_sources` is deliberately absent: citation snapshots are Week 4 (Plan §18). When it lands it
must keep NO foreign key to documents/document_versions/chunks (CLAUDE.md invariant #6).
`model_calls` holds operational metadata only; there is no column for a raw prompt or response.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-21
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _uuid_pk() -> sa.Column[sa.UUID]:
    return sa.Column(
        "id",
        postgresql.UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    )


def _created_at() -> sa.Column[sa.DateTime]:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )


def upgrade() -> None:
    op.create_table(
        "conversations",
        _uuid_pk(),
        sa.Column("user_id", sa.Text, nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )
    op.create_table(
        "turns",
        _uuid_pk(),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("sources_json", postgresql.JSONB),
        sa.Column("graph_status", sa.Text, nullable=False),
        sa.Column("retrieval_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_turns")),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_turns_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
    )
    op.create_table(
        "model_calls",
        _uuid_pk(),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("purpose", sa.Text, nullable=False),
        sa.Column("provider", sa.Text, nullable=False),
        sa.Column("model_name", sa.Text, nullable=False),
        sa.Column("prompt_version", sa.Text, nullable=False),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("error_code", sa.Text),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_calls")),
        sa.ForeignKeyConstraint(
            ["turn_id"],
            ["turns.id"],
            name=op.f("fk_model_calls_turn_id_turns"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "purpose IN ('plan', 'grade', 'answer', 'repair')", name=op.f("ck_model_calls_purpose")
        ),
        sa.CheckConstraint("status IN ('ok', 'error')", name=op.f("ck_model_calls_status")),
    )


def downgrade() -> None:
    op.drop_table("model_calls")
    op.drop_table("turns")
    op.drop_table("conversations")
