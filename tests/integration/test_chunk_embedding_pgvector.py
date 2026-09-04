from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.database.models.document import Document
from brd_knowledge.embeddings.base import EmbeddingConfiguration
from brd_knowledge.schemas.chunk import Chunk
from brd_knowledge.services.chunk_embedding_service import ChunkEmbeddingService

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.integration


class FakeProvider:
    configuration = EmbeddingConfiguration(
        provider="fake",
        model_name="fake-model",
        model_revision="revision-1",
        dimension=768,
        max_sequence_length=8192,
        pooling="cls",
        normalize_embeddings=True,
        output_dtype="float32",
        similarity_metric="cosine",
        trust_remote_code=False,
        query_prefix="",
        document_prefix="",
        preprocessing_version="1",
    )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 767 for _ in texts]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not configured")
def test_pgvector_sync_is_idempotent_and_deletes_stale_rows() -> None:
    assert TEST_DATABASE_URL is not None
    engine = create_engine(TEST_DATABASE_URL)
    document_id = f"integration-{uuid.uuid4()}"
    first = Chunk(
        chunk_id=f"{document_id}-1",
        document_id=document_id,
        content_type="prose",
        text="First requirement",
        page_start=1,
        page_end=1,
    )
    second = Chunk(
        chunk_id=f"{document_id}-2",
        document_id=document_id,
        content_type="table",
        text="Field | Value",
        page_start=2,
        page_end=2,
    )

    with Session(engine) as session:
        vector_extension = session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
        if vector_extension is None:
            pytest.skip("pgvector extension is not installed")
        session.add(
            Document(
                document_id=document_id,
                filename="integration.pdf",
                original_filename="integration.pdf",
                stored_filename="integration.pdf",
                stored_path="/tmp/integration.pdf",
                file_type="pdf",
                size_bytes=1,
                page_count=2,
                parsed_document_json={},
            )
        )
        session.commit()

        service = ChunkEmbeddingService(session, FakeProvider())
        first_result = service.synchronize(document_id, [first, second])
        second_result = service.synchronize(document_id, [first, second])
        delete_result = service.synchronize(document_id, [second])

        assert (first_result.embedded, first_result.skipped) == (2, 0)
        assert (second_result.embedded, second_result.skipped) == (0, 2)
        assert delete_result.deleted == 1
        rows = session.query(ChunkEmbedding).filter_by(document_id=document_id).all()
        assert [row.chunk_id for row in rows] == [second.chunk_id]

        session.query(Document).filter_by(document_id=document_id).delete()
        session.commit()
