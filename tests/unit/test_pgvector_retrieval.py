from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.embeddings.base import EmbeddingConfiguration
from brd_knowledge.retrieval import PgVectorRetriever


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

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0] + [0.0] * 767
        self.query_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("Retrieval must not call embed_documents.")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_calls.append(texts)
        return [self.vector]


class FakeExecution:
    def __init__(self, rows: list[tuple[ChunkEmbedding, float]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[ChunkEmbedding, float]]:
        return self._rows


class FakeSession:
    def __init__(self, rows: list[tuple[ChunkEmbedding, float]]) -> None:
        self._rows = rows
        self.statement: Any = None

    def execute(self, statement: Any) -> FakeExecution:
        self.statement = statement
        return FakeExecution(self._rows)


def stored_chunk() -> ChunkEmbedding:
    return ChunkEmbedding(
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="1) Scope",
        section_path=["1) Scope"],
        content_type="prose",
        text="The system shall keep an audit trail.",
        page_start=2,
        page_end=3,
        source_block_ids=["block-1"],
        source_table_ids=[],
        provenance=[{"document_id": "doc-1", "page_number": 2}],
        quality_notes=["reviewed"],
        content_hash="a" * 64,
        embedding_model="fake-model",
        embedding_revision="revision-1",
        embedding_config_hash=FakeProvider.configuration.config_hash,
        embedding_dimension=768,
        embedding=[1.0] + [0.0] * 767,
    )


def test_search_embeds_query_and_returns_ranked_traceable_metadata() -> None:
    provider = FakeProvider()
    session = FakeSession([(stored_chunk(), 0.25)])

    results = PgVectorRetriever(session, provider).search(
        "audit requirements", top_k=3, document_id="doc-1"  # type: ignore[arg-type]
    )

    assert provider.query_calls == [["audit requirements"]]
    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].cosine_distance == 0.25
    assert results[0].similarity == 0.75
    assert results[0].source_block_ids == ["block-1"]
    assert results[0].provenance == [{"document_id": "doc-1", "page_number": 2}]


def test_statement_uses_exact_cosine_search_and_embedding_identity_filters() -> None:
    provider = FakeProvider()
    retriever = PgVectorRetriever(FakeSession([]), provider)  # type: ignore[arg-type]

    statement = retriever._build_statement(provider.vector, 5, "doc-1")
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert "<=>" in sql
    assert "embedding_model" in sql
    assert "embedding_revision" in sql
    assert "embedding_config_hash" in sql
    assert "embedding_dimension" in sql
    assert "document_id" in sql
    assert "LIMIT" in sql


@pytest.mark.parametrize(("query", "top_k"), [("", 5), ("   ", 5), ("valid", 0)])
def test_rejects_empty_queries_and_invalid_top_k(query: str, top_k: int) -> None:
    provider = FakeProvider()
    retriever = PgVectorRetriever(FakeSession([]), provider)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        retriever.search(query, top_k=top_k)

    assert provider.query_calls == []


def test_rejects_invalid_query_vector_dimension() -> None:
    retriever = PgVectorRetriever(FakeSession([]), FakeProvider(vector=[1.0]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dimension 1"):
        retriever.search("valid query")


def test_rejects_unnormalized_query_vector() -> None:
    vector = [0.5] + [0.0] * 767
    retriever = PgVectorRetriever(FakeSession([]), FakeProvider(vector=vector))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="not normalized"):
        retriever.search("valid query")


def test_rejects_wrong_query_vector_count() -> None:
    class EmptyProvider(FakeProvider):
        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return []

    retriever = PgVectorRetriever(FakeSession([]), EmptyProvider())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="0 vectors for one query"):
        retriever.search("valid query")
