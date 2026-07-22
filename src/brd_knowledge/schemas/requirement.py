from pydantic import BaseModel, Field

from brd_knowledge.schemas.source import SourceReference


class ExtractedRequirement(BaseModel):
    requirement_id: str
    document_id: str
    text: str
    source: SourceReference
    requirement_type: str | None = None
    module: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
