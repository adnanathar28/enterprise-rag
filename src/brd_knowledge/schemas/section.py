from __future__ import annotations

from pydantic import BaseModel, Field

from brd_knowledge.schemas.document import DocumentBlock, TextBlock
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable


class DocumentSection(BaseModel):
    section_id: str
    title: str
    level: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source: SourceReference | None = None
    heading_block: TextBlock | None = None
    blocks: list[DocumentBlock] = Field(default_factory=list)
    paragraphs: list[TextBlock] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    child_sections: list[DocumentSection] = Field(default_factory=list)
