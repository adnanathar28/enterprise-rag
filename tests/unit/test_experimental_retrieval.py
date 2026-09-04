from __future__ import annotations

from typing import Any

from sqlalchemy.dialects import postgresql

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.retrieval.experimental import (
    RRF_K,
    PostgresLexicalRetriever,
    reciprocal_rank_fusion,
)
from brd_knowledge.schemas.experimental_retrieval import ExperimentalRetrievedChunk


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


def stored_chunk(chunk_id: str) -> ChunkEmbedding:
    return ChunkEmbedding(
        chunk_id=chunk_id,
        document_id="doc-1",
        section_path=["Requirements"],
        content_type="prose",
        text=f"Evidence {chunk_id}",
        page_start=1,
        page_end=1,
        source_block_ids=[f"block-{chunk_id}"],
        source_table_ids=[],
        provenance=[{"document_id": "doc-1", "page_number": 1}],
        quality_notes=[],
        content_hash="a" * 64,
        embedding_model="model",
        embedding_revision="revision",
        embedding_config_hash="config",
        embedding_dimension=768,
        embedding=[1.0] + [0.0] * 767,
    )


def ranked(chunk_id: str, rank: int, strategy: str) -> ExperimentalRetrievedChunk:
    return ExperimentalRetrievedChunk(
        rank=rank,
        score=1 / (rank + 1),
        strategy=strategy,  # type: ignore[arg-type]
        chunk_id=chunk_id,
        document_id="doc-1",
        section_path=["Requirements"],
        content_type="prose",
        text=chunk_id,
        page_start=1,
        page_end=1,
    )


def test_lexical_retriever_preserves_database_rank_order_and_metadata() -> None:
    session = FakeSession([(stored_chunk("b"), 0.8), (stored_chunk("a"), 0.4)])

    results = PostgresLexicalRetriever(session).search(
        "backup requirements", top_k=2, document_id="doc-1"  # type: ignore[arg-type]
    )

    assert [result.chunk_id for result in results] == ["b", "a"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.score for result in results] == [0.8, 0.4]
    assert results[0].source_block_ids == ["block-b"]


def test_lexical_statement_uses_required_postgresql_full_text_functions() -> None:
    retriever = PostgresLexicalRetriever(FakeSession([]))  # type: ignore[arg-type]

    statement = retriever._build_statement("backup requirements", 5, "doc-1")
    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "to_tsvector" in sql
    assert "plainto_tsquery" in sql
    assert "ts_rank_cd" in sql
    assert "@@" in sql
    assert "document_id" in sql


def test_rrf_merges_duplicates_and_uses_fixed_k_60() -> None:
    dense = [ranked("a", 1, "dense"), ranked("b", 2, "dense")]
    lexical = [ranked("b", 1, "lexical"), ranked("c", 2, "lexical")]

    results = reciprocal_rank_fusion(dense, lexical, top_k=3)

    assert RRF_K == 60
    assert [result.chunk_id for result in results] == ["b", "a", "c"]
    assert len({result.chunk_id for result in results}) == 3
    assert results[0].score == 1 / 62 + 1 / 61
    assert results[1].score == 1 / 61
    assert all(result.strategy == "hybrid_rrf" for result in results)


def test_rrf_ties_are_ordered_deterministically_by_chunk_id() -> None:
    dense = [ranked("b", 1, "dense")]
    lexical = [ranked("a", 1, "lexical")]

    first = reciprocal_rank_fusion(dense, lexical, top_k=2)
    second = reciprocal_rank_fusion(dense, lexical, top_k=2)

    assert [result.chunk_id for result in first] == ["a", "b"]
    assert first == second
