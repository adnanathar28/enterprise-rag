from __future__ import annotations

import math
from typing import cast

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.embeddings.base import EmbeddingProvider
from brd_knowledge.schemas.chunk import ChunkContentType
from brd_knowledge.schemas.retrieval import RetrievedChunk

PGVECTOR_DIMENSION = 768


class PgVectorRetriever:
    def __init__(self, session: Session, provider: EmbeddingProvider) -> None:
        self._session = session
        self._provider = provider
        if provider.configuration.dimension != PGVECTOR_DIMENSION:
            raise ValueError(
                f"Provider dimension is {provider.configuration.dimension}; "
                f"chunk_embeddings requires {PGVECTOR_DIMENSION}."
            )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        vectors = self._provider.embed_queries([query])
        if len(vectors) != 1:
            raise ValueError(f"Provider returned {len(vectors)} vectors for one query.")
        query_vector = vectors[0]
        self._validate_query_vector(query_vector)

        statement = self._build_statement(query_vector, top_k, document_id)
        rows = self._session.execute(statement).all()
        return [
            self._to_result(rank, row, distance)
            for rank, (row, distance) in enumerate(rows, 1)
        ]

    def _build_statement(
        self,
        query_vector: list[float],
        top_k: int,
        document_id: str | None,
    ) -> Select[tuple[ChunkEmbedding, float]]:
        configuration = self._provider.configuration
        distance = ChunkEmbedding.embedding.cosine_distance(query_vector)
        statement = select(ChunkEmbedding, distance.label("cosine_distance")).where(
            ChunkEmbedding.embedding_model == configuration.model_name,
            ChunkEmbedding.embedding_revision == configuration.model_revision,
            ChunkEmbedding.embedding_config_hash == configuration.config_hash,
            ChunkEmbedding.embedding_dimension == configuration.dimension,
        )
        if document_id is not None:
            statement = statement.where(ChunkEmbedding.document_id == document_id)
        return statement.order_by(distance.asc(), ChunkEmbedding.chunk_id.asc()).limit(top_k)

    def _validate_query_vector(self, vector: list[float]) -> None:
        dimension = self._provider.configuration.dimension
        if len(vector) != dimension:
            raise ValueError(f"Query vector has dimension {len(vector)}; expected {dimension}.")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Query vector contains a non-finite value.")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"Query vector is not normalized; L2 norm is {norm:.6f}.")

    @staticmethod
    def _to_result(rank: int, row: ChunkEmbedding, distance: float) -> RetrievedChunk:
        distance_value = float(distance)
        return RetrievedChunk(
            rank=rank,
            cosine_distance=distance_value,
            similarity=1.0 - distance_value,
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
