from pydantic import BaseModel, ConfigDict, Field, field_validator

from brd_knowledge.schemas.context import ConstructedContext
from brd_knowledge.schemas.source import SourceReference


class GroundedAnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    context: ConstructedContext
    max_output_tokens: int = Field(default=512, gt=0)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value


class ModelAnswerPayload(BaseModel):
    answer_text: str = Field(min_length=1)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool

    model_config = ConfigDict(extra="forbid")

    @field_validator("answer_text")
    @classmethod
    def reject_blank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Answer text must not be blank.")
        return value


class ResolvedCitation(BaseModel):
    evidence_id: str
    document_id: str
    chunk_id: str
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_block_ids: list[str] = Field(default_factory=list)
    source_table_ids: list[str] = Field(default_factory=list)
    provenance: list[SourceReference] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)


class GenerationMetadata(BaseModel):
    provider: str
    model: str
    revision: str | None = None
    max_output_tokens: int = Field(gt=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    model_version: str | None = None
    thinking_tokens: int | None = Field(default=None, ge=0)


class GroundedAnswer(BaseModel):
    answer_text: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    citations: list[ResolvedCitation] = Field(default_factory=list)
    insufficient_evidence: bool
    metadata: GenerationMetadata
