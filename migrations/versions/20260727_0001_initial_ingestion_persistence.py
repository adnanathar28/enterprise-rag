"""initial ingestion persistence

Revision ID: 20260727_0001
Revises:
Create Date: 2026-07-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260727_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("stored_filename", sa.String(length=512), nullable=False),
        sa.Column("stored_path", sa.String(length=1024), nullable=False),
        sa.Column("file_type", sa.String(length=32), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("parse_status", sa.String(length=64), nullable=True),
        sa.Column("parser_name", sa.String(length=128), nullable=True),
        sa.Column("parser_version", sa.String(length=128), nullable=True),
        sa.Column("parsed_document_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_documents_document_id"), "documents", ["document_id"], unique=True)

    op.create_table(
        "extraction_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_extraction_runs_document_id"),
        "extraction_runs",
        ["document_id"],
        unique=False,
    )

    op.create_table(
        "requirements",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("requirement_id", sa.String(length=128), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_requirements_document_id"),
        "requirements",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_requirements_requirement_id"),
        "requirements",
        ["requirement_id"],
        unique=True,
    )

    op.create_table(
        "sections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("section_id", sa.String(length=128), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_sections_document_id"), "sections", ["document_id"], unique=False)
    op.create_index(op.f("ix_sections_section_id"), "sections", ["section_id"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_sections_section_id"), table_name="sections")
    op.drop_index(op.f("ix_sections_document_id"), table_name="sections")
    op.drop_table("sections")

    op.drop_index(op.f("ix_requirements_requirement_id"), table_name="requirements")
    op.drop_index(op.f("ix_requirements_document_id"), table_name="requirements")
    op.drop_table("requirements")

    op.drop_index(op.f("ix_extraction_runs_document_id"), table_name="extraction_runs")
    op.drop_table("extraction_runs")

    op.drop_index(op.f("ix_documents_document_id"), table_name="documents")
    op.drop_table("documents")
