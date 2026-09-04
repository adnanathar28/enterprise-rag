from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from sqlalchemy.orm import Session

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.database.models.document import Document
from brd_knowledge.embeddings.base import EmbeddingProvider
from brd_knowledge.schemas.chunk import Chunk

PGVECTOR_DIMENSION = 768


@dataclass(frozen=True)
class EmbeddingSyncResult:
    total: int
    embedded: int
    updated: int
    skipped: int
    deleted: int
    model: str
    revision: str
    dimension: int


class ChunkEmbeddingService:
    def __init__(self, session: Session, provider: EmbeddingProvider) -> None:
        self._session = session
        self._provider = provider
        if provider.configuration.dimension != PGVECTOR_DIMENSION:
            raise ValueError(
                f"Provider dimension is {provider.configuration.dimension}; "
                f"chunk_embeddings requires {PGVECTOR_DIMENSION}."
            )

    def synchronize(self, document_id: str, chunks: list[Chunk]) -> EmbeddingSyncResult:
        self._validate_chunks(document_id, chunks)
        configuration = self._provider.configuration

        try:
            document_exists = (
                self._session.query(Document)
                .filter(Document.document_id == document_id)
                .one_or_none()
            )
            if document_exists is None:
                raise ValueError(
                    f"Document {document_id} is not persisted; ingest it before embedding chunks."
                )
            existing_rows = (
                self._session.query(ChunkEmbedding)
                .filter(ChunkEmbedding.document_id == document_id)
                .all()
            )
            existing_by_id = {row.chunk_id: row for row in existing_rows}
            incoming_ids = {chunk.chunk_id for chunk in chunks}
            stale_rows = [row for row in existing_rows if row.chunk_id not in incoming_ids]

            new_chunks: list[Chunk] = []
            changed_chunks: list[Chunk] = []
            skipped = 0
            for chunk in chunks:
                row = existing_by_id.get(chunk.chunk_id)
                if row is None:
                    new_chunks.append(chunk)
                elif self._requires_embedding(row, chunk):
                    changed_chunks.append(chunk)
                else:
                    self._populate_metadata(row, chunk)
                    skipped += 1

            pending = new_chunks + changed_chunks
            vectors = (
                self._provider.embed_documents([chunk.text for chunk in pending]) if pending else []
            )
            self._validate_vectors(vectors, len(pending))

            for chunk, vector in zip(pending, vectors, strict=True):
                row = existing_by_id.get(chunk.chunk_id)
                if row is None:
                    row = ChunkEmbedding(chunk_id=chunk.chunk_id, document_id=document_id)
                    self._session.add(row)
                self._populate(row, chunk, vector)

            for row in stale_rows:
                self._session.delete(row)

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return EmbeddingSyncResult(
            total=len(chunks),
            embedded=len(new_chunks),
            updated=len(changed_chunks),
            skipped=skipped,
            deleted=len(stale_rows),
            model=configuration.model_name,
            revision=configuration.model_revision,
            dimension=configuration.dimension,
        )

    def _requires_embedding(self, row: ChunkEmbedding, chunk: Chunk) -> bool:
        configuration = self._provider.configuration
        return (
            row.content_hash != self._content_hash(chunk.text)
            or row.embedding_model != configuration.model_name
            or row.embedding_revision != configuration.model_revision
            or row.embedding_config_hash != configuration.config_hash
            or row.embedding_dimension != configuration.dimension
        )

    def _populate(self, row: ChunkEmbedding, chunk: Chunk, vector: list[float]) -> None:
        configuration = self._provider.configuration
        self._populate_metadata(row, chunk)
        row.content_hash = self._content_hash(chunk.text)
        row.embedding_model = configuration.model_name
        row.embedding_revision = configuration.model_revision
        row.embedding_config_hash = configuration.config_hash
        row.embedding_dimension = configuration.dimension
        row.embedding = vector

    def _populate_metadata(self, row: ChunkEmbedding, chunk: Chunk) -> None:
        row.section_id = chunk.section_id
        row.section_title = chunk.section_title
        row.section_path = chunk.section_path
        row.content_type = chunk.content_type
        row.text = chunk.text
        row.page_start = chunk.page_start
        row.page_end = chunk.page_end
        row.source_block_ids = chunk.source_block_ids
        row.source_table_ids = chunk.source_table_ids
        row.provenance = [source.model_dump(mode="json") for source in chunk.provenance]
        row.quality_notes = chunk.quality_notes

    def _validate_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        chunk_ids: set[str] = set()
        for chunk in chunks:
            if chunk.document_id != document_id:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} belongs to {chunk.document_id}, not {document_id}."
                )
            if chunk.chunk_id in chunk_ids:
                raise ValueError(f"Duplicate chunk_id in synchronization input: {chunk.chunk_id}")
            chunk_ids.add(chunk.chunk_id)

    def _validate_vectors(self, vectors: list[list[float]], expected_count: int) -> None:
        dimension = self._provider.configuration.dimension
        if len(vectors) != expected_count:
            raise ValueError(
                f"Provider returned {len(vectors)} vectors for {expected_count} texts."
            )
        for index, vector in enumerate(vectors):
            if len(vector) != dimension:
                raise ValueError(
                    f"Provider vector {index} has dimension {len(vector)}; expected {dimension}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError(f"Provider vector {index} contains a non-finite value.")

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
