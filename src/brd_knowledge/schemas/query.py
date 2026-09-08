from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from brd_knowledge.schemas.context import ConstructedContext
from brd_knowledge.schemas.generation import GroundedAnswer
from brd_knowledge.schemas.retrieval import RetrievedChunk

LLMProviderName = Literal["gemini", "local_qwen"]


class DocumentQuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    provider: LLMProviderName | None = None
    model: str | None = Field(default=None, min_length=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Model must not be blank.")
        return value


class DocumentQuestionResponse(BaseModel):
    document_id: str
    question: str
    answer: GroundedAnswer
    context: ConstructedContext
    retrieved_chunks: list[RetrievedChunk]
    elapsed_seconds: float = Field(ge=0)


class ProviderCapability(BaseModel):
    provider: LLMProviderName
    label: str
    model: str
    configured: bool
    is_default: bool


class ApplicationCapabilities(BaseModel):
    providers: list[ProviderCapability]
    allowed_document_extensions: list[str]
    max_upload_size_bytes: int = Field(gt=0)
