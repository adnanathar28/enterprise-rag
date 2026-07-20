from pydantic import BaseModel, Field

from brd_knowledge.schemas.table import BoundingBox


class SourceReference(BaseModel):
    document_id: str
    page_number: int = Field(ge=1)
    section_id: str | None = None
    table_id: str | None = None
    block_id: str | None = None
    text_excerpt: str | None = None
    bounding_box: BoundingBox | None = None


class ExtractedRequirement(BaseModel):
    requirement_id: str
    document_id: str
    text: str
    source: SourceReference
    requirement_type: str | None = None
    module: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
