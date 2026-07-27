from typing import Literal

from pydantic import BaseModel, Field

ExtractionContentType = Literal["paragraph", "list_item", "section_header", "table_row"]


class ExtractionSourceRef(BaseModel):
    document_id: str
    page_number: int | None = Field(default=None, ge=1)
    section_id: str | None = None
    block_id: str | None = None
    table_id: str | None = None
    cell_ids: list[str] = Field(default_factory=list)
    reading_order_index: int | None = Field(default=None, ge=0)


class ExtractionUnit(BaseModel):
    unit_id: str
    document_id: str
    page_number: int = Field(ge=1)
    content_type: ExtractionContentType
    text: str
    source_refs: list[ExtractionSourceRef]
    section_id: str | None = None
    section_title: str | None = None
    reading_order_index: int | None = Field(default=None, ge=0)
    quality_notes: list[str] = Field(default_factory=list)
