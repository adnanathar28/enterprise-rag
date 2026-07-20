from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from brd_knowledge.schemas.table import BoundingBox, ParsedTable

if TYPE_CHECKING:
    from brd_knowledge.schemas.section import DocumentSection


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


class TextBlock(BaseModel):
    block_id: str
    page_number: int = Field(ge=1)
    text: str
    block_type: str = "paragraph"
    bounding_box: BoundingBox | None = None


class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    pages: list[int] = Field(default_factory=list)
    sections: list["DocumentSection"] = Field(default_factory=list)
    paragraphs: list[TextBlock] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)


from brd_knowledge.schemas.section import DocumentSection  # noqa: E402

ParsedDocument.model_rebuild()
