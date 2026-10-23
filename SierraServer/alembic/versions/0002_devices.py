"""add privacy-bounded Sierra device profiles

Revision ID: 0002_devices
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_devices"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("install_id", sa.String(64), nullable=False),
        sa.Column("manufacturer", sa.String(80), nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column("device_name", sa.String(120)),
        sa.Column("android_release", sa.String(32), nullable=False),
        sa.Column("sdk_int", sa.Integer(), nullable=False),
        sa.Column("app_version", sa.String(40), nullable=False),
        sa.Column("app_build", sa.Integer(), nullable=False),
        sa.Column("locale", sa.String(40), nullable=False),
        sa.Column("timezone", sa.String(80), nullable=False),
        sa.Column("screen_width_px", sa.Integer()),
        sa.Column("screen_height_px", sa.Integer()),
        sa.Column("density_dpi", sa.Integer()),
        sa.Column("is_emulator", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("sierra_beta_version", sa.String(40)),
        sa.Column("camera_capabilities", sa.JSON()),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_ip", sa.String(64)),
        sa.Column("last_user_agent", sa.String(300)),
        sa.UniqueConstraint("user_id", "install_id", name="uq_devices_user_install"),
    )
    op.create_index("ix_devices_user_id", "devices", ["user_id"])
    op.create_index("ix_devices_install_id", "devices", ["install_id"])


def downgrade() -> None:
    op.drop_index("ix_devices_install_id", table_name="devices")
    op.drop_index("ix_devices_user_id", table_name="devices")
    op.drop_table("devices")
