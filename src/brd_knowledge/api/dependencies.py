from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from brd_knowledge.core.config import Settings, get_settings
from brd_knowledge.database.session import get_db_session
from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.parsing.docling_parser import DoclingDocumentParser
from brd_knowledge.retrieval import PgVectorRetriever
from brd_knowledge.services.chunk_embedding_service import ChunkEmbeddingService
from brd_knowledge.services.document_indexing_service import DocumentIndexingService
from brd_knowledge.services.document_persistence_service import DocumentPersistenceService
from brd_knowledge.services.file_intake_service import FileIntakeService
from brd_knowledge.services.ingestion_service import IngestionService
from brd_knowledge.services.question_answering_service import QuestionAnsweringService


def settings_dependency() -> Settings:
    return get_settings()


def file_intake_service_dependency() -> FileIntakeService:
    settings = get_settings()
    return FileIntakeService(
        storage_dir=settings.source_storage_dir,
        allowed_extensions=settings.allowed_document_extensions,
        max_upload_size_bytes=settings.max_upload_size_bytes,
    )


def ingestion_service_dependency() -> IngestionService:
    return IngestionService(parsers=[DoclingDocumentParser()])


def document_persistence_service_dependency(
    session: Annotated[Session, Depends(get_db_session)],
) -> DocumentPersistenceService:
    return DocumentPersistenceService(session)


@lru_cache
def embedding_provider_dependency() -> GteModernBertEmbeddingProvider:
    settings = get_settings()
    return GteModernBertEmbeddingProvider(
        model_name=settings.embedding_model_name,
        model_revision=settings.embedding_model_revision,
        dimension=settings.embedding_dimension,
        max_sequence_length=settings.embedding_max_sequence_length,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        preprocessing_version=settings.embedding_preprocessing_version,
    )


def question_answering_service_dependency(
    session: Annotated[Session, Depends(get_db_session)],
    embedding_provider: Annotated[
        GteModernBertEmbeddingProvider,
        Depends(embedding_provider_dependency),
    ],
) -> QuestionAnsweringService:
    return QuestionAnsweringService(
        PgVectorRetriever(session, embedding_provider),
        get_settings(),
    )


def document_indexing_service_dependency(
    session: Annotated[Session, Depends(get_db_session)],
    embedding_provider: Annotated[
        GteModernBertEmbeddingProvider,
        Depends(embedding_provider_dependency),
    ],
) -> DocumentIndexingService:
    return DocumentIndexingService(
        DocumentPersistenceService(session),
        ChunkEmbeddingService(session, embedding_provider),
        embedding_provider.configuration,
    )
