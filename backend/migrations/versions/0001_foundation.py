"""Create Room3D foundation tables."""
from alembic import op
import sqlalchemy as sa

revision = "0001_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("users", sa.Column("id", sa.String(36), primary_key=True), sa.Column("email", sa.String(320), nullable=False), sa.Column("password_hash", sa.Text(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table("projects", sa.Column("id", sa.String(36), primary_key=True), sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True), sa.Column("name", sa.String(255), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("active_version_id", sa.String(36)), sa.Column("video_relpath", sa.Text()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_projects_owner_id", "projects", ["owner_id"])
    op.create_table("scene_versions", sa.Column("id", sa.String(36), primary_key=True), sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id"), nullable=False), sa.Column("parent_version_id", sa.String(36)), sa.Column("label", sa.String(255), nullable=False), sa.Column("manifest_relpath", sa.Text()), sa.Column("spz_relpath", sa.Text()), sa.Column("ply_relpath", sa.Text()), sa.Column("collision_glb_relpath", sa.Text()), sa.Column("metrics_json", sa.Text()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_scene_versions_project_id", "scene_versions", ["project_id"])
    op.create_table("edit_sessions", sa.Column("id", sa.String(36), primary_key=True), sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id"), nullable=False), sa.Column("base_version_id", sa.String(36)), sa.Column("target_version_id", sa.String(36)), sa.Column("ref_frame_idx", sa.Integer()), sa.Column("prompt", sa.Text()), sa.Column("points_json", sa.Text()), sa.Column("masks_summary_json", sa.Text()), sa.Column("chosen_seed_id", sa.String(64)), sa.Column("status", sa.String(32), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_edit_sessions_project_id", "edit_sessions", ["project_id"])
    op.create_table("tasks", sa.Column("id", sa.String(36), primary_key=True), sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id"), nullable=False), sa.Column("edit_session_id", sa.String(36), sa.ForeignKey("edit_sessions.id")), sa.Column("task_type", sa.String(64), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("current_stage", sa.String(64)), sa.Column("progress_pct", sa.Float(), nullable=False), sa.Column("message", sa.Text()), sa.Column("payload_json", sa.Text()), sa.Column("result_json", sa.Text()), sa.Column("active_attempt_id", sa.String(36)), sa.Column("retry_count", sa.Integer(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_tasks_project_id", "tasks", ["project_id"])
    op.create_index("ix_tasks_status", "tasks", ["status"])
    op.create_table("task_attempts", sa.Column("id", sa.String(36), primary_key=True), sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id"), nullable=False), sa.Column("worker_id", sa.String(255), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("heartbeat_at", sa.DateTime(timezone=True)), sa.Column("finished_at", sa.DateTime(timezone=True)), sa.Column("error_code", sa.String(64)), sa.Column("error_message", sa.Text()))
    op.create_index("ix_task_attempts_task_id", "task_attempts", ["task_id"])


def downgrade() -> None:
    op.drop_table("task_attempts")
    op.drop_table("tasks")
    op.drop_table("edit_sessions")
    op.drop_table("scene_versions")
    op.drop_table("projects")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
