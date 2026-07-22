from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from brd_knowledge.schemas.source import BoundingBox, SourceReference
from brd_knowledge.schemas.table import ParsedTable

if TYPE_CHECKING:
    from brd_knowledge.schemas.section import DocumentSection

ParseStatus = Literal["not_started", "success", "partial_success", "failed", "skipped"]
DiagnosticSeverity = Literal["info", "warning", "error"]


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    source_path: Path | None = None
    file_type: str
    page_count: int = Field(ge=0)
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    parser_name: str | None = None
    parser_version: str | None = None


class ParserDiagnostic(BaseModel):
    severity: DiagnosticSeverity
    message: str
    parser_name: str | None = None
    parser_version: str | None = None
    stage: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    error_type: str | None = None


class ParserMetadata(BaseModel):
    parser_name: str
    parser_version: str | None = None
    parse_status: ParseStatus = "not_started"
    parse_strategy: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    diagnostics: list[ParserDiagnostic] = Field(default_factory=list)


class DocumentBlock(BaseModel):
    block_id: str
    page_number: int = Field(ge=1)
    text: str
    block_type: str = "paragraph"
    reading_order_index: int | None = Field(default=None, ge=0)
    bounding_box: BoundingBox | None = None
    source: SourceReference | None = None


class TextBlock(DocumentBlock):
    pass


class ParsedImage(BaseModel):
    image_id: str
    page_number: int = Field(ge=1)
    reading_order_index: int | None = Field(default=None, ge=0)
    caption: str | None = None
    bounding_box: BoundingBox | None = None
    source: SourceReference | None = None
    quality_notes: list[str] = Field(default_factory=list)


class Page(BaseModel):
    page_number: int = Field(ge=1)
    width: float | None = Field(default=None, ge=0)
    height: float | None = Field(default=None, ge=0)
    parse_status: ParseStatus = "not_started"
    parser_name: str | None = None
    parser_version: str | None = None
    diagnostics: list[ParserDiagnostic] = Field(default_factory=list)
    blocks: list[DocumentBlock] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    images: list[ParsedImage] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    parser_metadata: ParserMetadata | None = None
    pages: list[Page] = Field(default_factory=list)
    sections: list["DocumentSection"] = Field(default_factory=list)
    blocks: list[DocumentBlock] = Field(default_factory=list)
    paragraphs: list[TextBlock] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    images: list[ParsedImage] = Field(default_factory=list)
    diagnostics: list[ParserDiagnostic] = Field(default_factory=list)


from brd_knowledge.schemas.section import DocumentSection  # noqa: E402

ParsedDocument.model_rebuild()
