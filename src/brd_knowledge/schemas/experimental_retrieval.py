from typing import Any, Literal

from pydantic import BaseModel, Field

from brd_knowledge.schemas.chunk import ChunkContentType

ExperimentalStrategy = Literal["dense", "lexical", "hybrid_rrf"]


class ExperimentalRetrievedChunk(BaseModel):
    rank: int = Field(ge=1)
    score: float
    strategy: ExperimentalStrategy
    chunk_id: str
    document_id: str
    section_id: str | None = None
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    content_type: ChunkContentType
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_block_ids: list[str] = Field(default_factory=list)
    source_table_ids: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
