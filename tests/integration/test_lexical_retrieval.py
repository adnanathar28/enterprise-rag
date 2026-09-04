from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.database.models.document import Document
from brd_knowledge.retrieval import PostgresLexicalRetriever

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.integration


def row(document_id: str, chunk_id: str, text: str) -> ChunkEmbedding:
    return ChunkEmbedding(
        chunk_id=chunk_id,
        document_id=document_id,
        section_path=["Requirements"],
        content_type="prose",
        text=text,
        page_start=1,
        page_end=1,
        source_block_ids=[f"block-{chunk_id}"],
        source_table_ids=[],
        provenance=[{"document_id": document_id, "page_number": 1}],
        quality_notes=[],
        content_hash="a" * 64,
        embedding_model="test",
        embedding_revision="test",
        embedding_config_hash="test",
        embedding_dimension=768,
        embedding=[1.0] + [0.0] * 767,
    )


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not configured")
def test_postgresql_lexical_search_ranks_literal_evidence() -> None:
    assert TEST_DATABASE_URL is not None
    engine = create_engine(TEST_DATABASE_URL)
    document_id = f"lexical-{uuid.uuid4()}"
    with Session(engine) as session:
        session.add(
            Document(
                document_id=document_id,
                filename="lexical.pdf",
                original_filename="lexical.pdf",
                stored_filename="lexical.pdf",
                stored_path="/tmp/lexical.pdf",
                file_type="pdf",
                size_bytes=1,
                page_count=1,
                parsed_document_json={},
            )
        )
        session.commit()
        session.add_all(
            [
                row(document_id, f"{document_id}-relevant", "Daily backup completes at midnight"),
                row(document_id, f"{document_id}-other", "Inventory user interface guidance"),
            ]
        )
        session.commit()

        results = PostgresLexicalRetriever(session).search(
            "midnight backup", top_k=5, document_id=document_id
        )

        assert [result.text for result in results] == ["Daily backup completes at midnight"]
        assert results[0].score > 0
        session.query(Document).filter_by(document_id=document_id).delete()
        session.commit()
