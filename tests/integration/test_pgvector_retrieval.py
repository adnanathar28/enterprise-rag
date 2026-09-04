from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.database.models.document import Document
from brd_knowledge.embeddings.base import EmbeddingConfiguration
from brd_knowledge.retrieval import PgVectorRetriever

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.integration


class FakeProvider:
    configuration = EmbeddingConfiguration(
        provider="fake-retrieval",
        model_name="fake-retrieval-model",
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
        raise AssertionError("Not used by retrieval integration test.")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 767 for _ in texts]


def embedding_row(
    document_id: str,
    chunk_id: str,
    text: str,
    vector: list[float],
    *,
    config_hash: str,
) -> ChunkEmbedding:
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
        embedding_model=FakeProvider.configuration.model_name,
        embedding_revision=FakeProvider.configuration.model_revision,
        embedding_config_hash=config_hash,
        embedding_dimension=768,
        embedding=vector,
    )


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not configured")
def test_exact_cosine_search_ranks_results_and_excludes_incompatible_space() -> None:
    assert TEST_DATABASE_URL is not None
    engine = create_engine(TEST_DATABASE_URL)
    document_id = f"retrieval-{uuid.uuid4()}"
    provider = FakeProvider()

    with Session(engine) as session:
        session.add(
            Document(
                document_id=document_id,
                filename="retrieval.pdf",
                original_filename="retrieval.pdf",
                stored_filename="retrieval.pdf",
                stored_path="/tmp/retrieval.pdf",
                file_type="pdf",
                size_bytes=1,
                page_count=1,
                parsed_document_json={},
            )
        )
        session.commit()
        session.add_all(
            [
                embedding_row(
                    document_id,
                    f"{document_id}-exact",
                    "Exact",
                    [1.0] + [0.0] * 767,
                    config_hash=provider.configuration.config_hash,
                ),
                embedding_row(
                    document_id,
                    f"{document_id}-orthogonal",
                    "Orthogonal",
                    [0.0, 1.0] + [0.0] * 766,
                    config_hash=provider.configuration.config_hash,
                ),
                embedding_row(
                    document_id,
                    f"{document_id}-incompatible",
                    "Wrong space",
                    [1.0] + [0.0] * 767,
                    config_hash="different-config",
                ),
            ]
        )
        session.commit()

        results = PgVectorRetriever(session, provider).search(
            "query", top_k=5, document_id=document_id
        )

        assert [result.text for result in results] == ["Exact", "Orthogonal"]
        assert results[0].similarity == pytest.approx(1.0)
        assert results[1].similarity == pytest.approx(0.0)
        assert [result.rank for result in results] == [1, 2]

        session.query(Document).filter_by(document_id=document_id).delete()
        session.commit()
