from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.schemas.chunk import ChunkContentType
from brd_knowledge.schemas.experimental_retrieval import ExperimentalRetrievedChunk
from brd_knowledge.schemas.retrieval import RetrievedChunk

RRF_K = 60


class DenseRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]: ...


class ExperimentalRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[ExperimentalRetrievedChunk]: ...


class DenseExperimentRetriever:
    def __init__(self, retriever: DenseRetriever) -> None:
        self._retriever = retriever

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[ExperimentalRetrievedChunk]:
        results = self._retriever.search(query, top_k=top_k, document_id=document_id)
        return [self._from_dense(result) for result in results]

    @staticmethod
    def _from_dense(result: RetrievedChunk) -> ExperimentalRetrievedChunk:
        return ExperimentalRetrievedChunk(
            rank=result.rank,
            score=result.similarity,
            strategy="dense",
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            section_id=result.section_id,
            section_title=result.section_title,
            section_path=result.section_path,
            content_type=result.content_type,
            text=result.text,
            page_start=result.page_start,
            page_end=result.page_end,
            source_block_ids=result.source_block_ids,
            source_table_ids=result.source_table_ids,
            provenance=result.provenance,
            quality_notes=result.quality_notes,
        )


class PostgresLexicalRetriever:
    def __init__(self, session: Session) -> None:
        self._session = session

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[ExperimentalRetrievedChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        statement = self._build_statement(query, top_k, document_id)
        rows = self._session.execute(statement).all()
        return [
            self._to_result(rank, row, score)
            for rank, (row, score) in enumerate(rows, start=1)
        ]

    def _build_statement(
        self,
        query: str,
        top_k: int,
        document_id: str | None,
    ) -> Select[tuple[ChunkEmbedding, float]]:
        text_vector = func.to_tsvector("english", ChunkEmbedding.text)
        text_query = func.plainto_tsquery("english", query)
        score = func.ts_rank_cd(text_vector, text_query)
        statement = select(ChunkEmbedding, score.label("lexical_score")).where(
            text_vector.op("@@")(text_query)
        )
        if document_id is not None:
            statement = statement.where(ChunkEmbedding.document_id == document_id)
        return statement.order_by(score.desc(), ChunkEmbedding.chunk_id.asc()).limit(top_k)

    @staticmethod
    def _to_result(
        rank: int,
        row: ChunkEmbedding,
        score: float,
    ) -> ExperimentalRetrievedChunk:
        return ExperimentalRetrievedChunk(
            rank=rank,
            score=float(score),
            strategy="lexical",
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            section_id=row.section_id,
            section_title=row.section_title,
            section_path=row.section_path,
            content_type=cast(ChunkContentType, row.content_type),
            text=row.text,
            page_start=row.page_start,
            page_end=row.page_end,
            source_block_ids=row.source_block_ids,
            source_table_ids=row.source_table_ids,
            provenance=row.provenance,
            quality_notes=row.quality_notes,
        )


def reciprocal_rank_fusion(
    dense: Sequence[ExperimentalRetrievedChunk],
    lexical: Sequence[ExperimentalRetrievedChunk],
    *,
    top_k: int,
) -> list[ExperimentalRetrievedChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    scores: dict[str, float] = {}
    chunks: dict[str, ExperimentalRetrievedChunk] = {}
    for ranking in (dense, lexical):
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        chunks[chunk_id].model_copy(
            update={
                "rank": rank,
                "score": scores[chunk_id],
                "strategy": "hybrid_rrf",
            }
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


class HybridRrfRetriever:
    def __init__(
        self,
        dense: ExperimentalRetriever,
        lexical: ExperimentalRetriever,
    ) -> None:
        self._dense = dense
        self._lexical = lexical

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[ExperimentalRetrievedChunk]:
        dense = self._dense.search(query, top_k=top_k, document_id=document_id)
        lexical = self._lexical.search(query, top_k=top_k, document_id=document_id)
        return reciprocal_rank_fusion(dense, lexical, top_k=top_k)
