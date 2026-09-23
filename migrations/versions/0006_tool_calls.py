"""tool_calls: operational metadata for subscription-tool calls (Plan §10, §17, §18 Tuần 7)

Structural twin of `model_calls` (0003): no column holds the raw `subscription_id`/`customer_id`
value, only `identifier_kind` (which kind was used). `turn_id` cascades like `model_calls.turn_id`
-- this table is pure observability, never a citation source (turn_sources/invariant #6 is
unrelated and untouched by this migration).

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-23
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tool_calls",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tool_name", sa.Text, nullable=False),
        sa.Column("identifier_kind", sa.Text, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("error_code", sa.Text),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_calls")),
        sa.ForeignKeyConstraint(
            ["turn_id"],
            ["turns.id"],
            name=op.f("fk_tool_calls_turn_id_turns"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "tool_name IN ('get_subscription')", name=op.f("ck_tool_calls_tool_name")
        ),
        sa.CheckConstraint(
            "identifier_kind IN ('subscription_id', 'customer_id')",
            name=op.f("ck_tool_calls_identifier_kind"),
        ),
        sa.CheckConstraint("status IN ('ok', 'error')", name=op.f("ck_tool_calls_status")),
    )


def downgrade() -> None:
    op.drop_table("tool_calls")
