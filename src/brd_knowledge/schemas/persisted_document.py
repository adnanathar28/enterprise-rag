from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from brd_knowledge.database.models.document import Document


class DocumentIndexingSummary(BaseModel):
    status: Literal["not_indexed", "ready", "needs_reindex"]
    compatible_chunk_count: int = Field(ge=0)
    total_chunk_count: int = Field(ge=0)


class PersistedDocumentSummary(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    stored_filename: str
    file_type: str
    size_bytes: int
    page_count: int
    parse_status: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    created_at: datetime
    indexing: DocumentIndexingSummary

    @classmethod
    def from_model(
        cls,
        document: Document,
        indexing: DocumentIndexingSummary,
    ) -> "PersistedDocumentSummary":
        return cls(
            document_id=document.document_id,
            filename=document.filename,
            original_filename=document.original_filename,
            stored_filename=document.stored_filename,
            file_type=document.file_type,
            size_bytes=document.size_bytes,
            page_count=document.page_count,
            parse_status=document.parse_status,
            parser_name=document.parser_name,
            parser_version=document.parser_version,
            created_at=document.created_at,
            indexing=indexing,
        )


class PersistedParsedDocument(BaseModel):
    document_id: str
    parsed_document: dict[str, Any]
