"""add chunk embeddings with pgvector

Revision ID: 20260904_0002
Revises: 20260727_0001
Create Date: 2026-09-04
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "20260904_0002"
down_revision = "20260727_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("section_id", sa.String(length=128), nullable=True),
        sa.Column("section_title", sa.String(length=512), nullable=True),
        sa.Column(
            "section_path",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("content_type", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column(
            "source_block_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "source_table_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("quality_notes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=256), nullable=False),
        sa.Column("embedding_revision", sa.String(length=128), nullable=False),
        sa.Column("embedding_config_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id", name="uq_chunk_embeddings_chunk_id"),
    )
    op.create_index(
        op.f("ix_chunk_embeddings_content_type"),
        "chunk_embeddings",
        ["content_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chunk_embeddings_document_id"),
        "chunk_embeddings",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chunk_embeddings_section_id"),
        "chunk_embeddings",
        ["section_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_chunk_embeddings_section_id"), table_name="chunk_embeddings")
    op.drop_index(op.f("ix_chunk_embeddings_document_id"), table_name="chunk_embeddings")
    op.drop_index(op.f("ix_chunk_embeddings_content_type"), table_name="chunk_embeddings")
    op.drop_table("chunk_embeddings")
