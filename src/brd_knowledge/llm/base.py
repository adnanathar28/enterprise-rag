from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMConfiguration(BaseModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str | None = None
    context_window_tokens: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    structured_output: bool = True
    timeout_seconds: float = Field(default=60.0, gt=0)
    safety_margin_tokens: int = Field(default=128, ge=0)


class LLMGenerationRequest(BaseModel):
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    response_schema: dict[str, Any]


class LLMGenerationResponse(BaseModel):
    payload: dict[str, Any]
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    request_id: str | None = None
    model_version: str | None = None
    thinking_tokens: int | None = Field(default=None, ge=0)


class LLMProvider(ABC):
    def close(self) -> None:
        """Release provider resources when present."""
        return None

    @property
    @abstractmethod
    def configuration(self) -> LLMConfiguration:
        """Return the pinned provider and model configuration."""

    @abstractmethod
    def count_tokens(self, request: LLMGenerationRequest) -> int:
        """Count the complete structured request using provider-supported token counting."""

    @abstractmethod
    def generate_structured(self, request: LLMGenerationRequest) -> LLMGenerationResponse:
        """Generate one schema-constrained response."""
