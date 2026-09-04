from typing import Literal

from pydantic import BaseModel, Field

from brd_knowledge.schemas.chunk import ChunkContentType
from brd_knowledge.schemas.retrieval import RetrievedChunk
from brd_knowledge.schemas.source import SourceReference

ContextExclusionReason = Literal["duplicate", "empty", "budget_exceeded"]


class ContextBuildRequest(BaseModel):
    retrieved_chunks: list[RetrievedChunk]
    max_characters: int = Field(gt=0)


class ContextEvidence(BaseModel):
    evidence_id: str
    retrieval_rank: int = Field(ge=1)
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
    provenance: list[SourceReference] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
    similarity: float
    rendered_characters: int = Field(ge=0)


class ContextExclusion(BaseModel):
    chunk_id: str
    retrieval_rank: int = Field(ge=1)
    reason: ContextExclusionReason
    rendered_characters: int | None = Field(default=None, ge=0)
    required_characters: int | None = Field(default=None, ge=0)
    remaining_characters: int | None = Field(default=None, ge=0)
    would_fit_remaining_budget: bool | None = None


class ConstructedContext(BaseModel):
    evidence: list[ContextEvidence]
    rendered_text: str
    max_characters: int = Field(gt=0)
    used_characters: int = Field(ge=0)
    unused_characters: int = Field(ge=0)
    exclusions: list[ContextExclusion] = Field(default_factory=list)
