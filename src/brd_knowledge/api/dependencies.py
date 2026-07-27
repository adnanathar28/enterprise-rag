from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from brd_knowledge.core.config import Settings, get_settings
from brd_knowledge.database.session import get_db_session
from brd_knowledge.parsing.docling_parser import DoclingDocumentParser
from brd_knowledge.services.document_persistence_service import DocumentPersistenceService
from brd_knowledge.services.file_intake_service import FileIntakeService
from brd_knowledge.services.ingestion_service import IngestionService


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
