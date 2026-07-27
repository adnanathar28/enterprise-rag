from typing import Any

from sqlalchemy.orm import Session

from brd_knowledge.database.models.document import Document
from brd_knowledge.schemas.ingestion import IngestionResult
from brd_knowledge.schemas.source_file import StoredSourceFile


class DocumentPersistenceService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_ingestion_result(
        self,
        stored_file: StoredSourceFile,
        ingestion_result: IngestionResult,
    ) -> Document:
        parsed_document = ingestion_result.parsed_document
        parser_metadata = parsed_document.parser_metadata
        document = Document(
            document_id=ingestion_result.document_id,
            filename=stored_file.original_filename,
            original_filename=stored_file.original_filename,
            stored_filename=stored_file.stored_filename,
            stored_path=str(stored_file.stored_path),
            file_type=stored_file.extension.lstrip("."),
            size_bytes=stored_file.size_bytes,
            page_count=ingestion_result.page_count,
            parse_status=ingestion_result.parse_status,
            parser_name=parsed_document.metadata.parser_name,
            parser_version=(
                parser_metadata.parser_version
                if parser_metadata is not None
                else parsed_document.metadata.parser_version
            ),
            parsed_document_json=parsed_document.model_dump(mode="json"),
        )
        self._session.add(document)
        self._session.commit()
        self._session.refresh(document)
        return document

    def get_parsed_document_json(self, document_id: str) -> dict[str, Any] | None:
        document = (
            self._session.query(Document).filter(Document.document_id == document_id).one_or_none()
        )
        if document is None:
            return None
        return document.parsed_document_json
