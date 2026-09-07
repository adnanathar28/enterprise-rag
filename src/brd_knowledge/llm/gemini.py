from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from google import genai
from google.genai import errors, types
from pydantic import Field, SecretStr, ValidationError

from brd_knowledge.core.exceptions import (
    GenerationBlockedError,
    GenerationProviderError,
    IncompleteGenerationError,
    MalformedGenerationResponse,
    PromptBudgetExceeded,
)
from brd_knowledge.llm.base import (
    LLMConfiguration,
    LLMGenerationRequest,
    LLMGenerationResponse,
    LLMProvider,
)


class GeminiConfiguration(LLMConfiguration):
    provider: Literal["gemini"] = "gemini"
    model: Literal["gemini-3.1-flash-lite"] = "gemini-3.1-flash-lite"
    context_window_tokens: int = Field(default=1_048_576, gt=0, le=1_048_576)
    max_output_tokens: int = Field(default=65_536, gt=0, le=65_536)
    temperature: float = Field(default=1.0, ge=0, le=2)
    thinking_level: Literal["minimal"] = "minimal"
    structured_output: Literal[True] = True
    api_version: Literal["v1beta"] = "v1beta"
    attempts: Literal[1] = 1


class GeminiLLMProvider(LLMProvider):
    """One stateless Developer API generation, with no SDK retries or tools."""

    def __init__(
        self,
        api_key: SecretStr,
        configuration: GeminiConfiguration | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key.get_secret_value().strip():
            raise GenerationProviderError("GEMINI_API_KEY must be configured.")
        self._configuration = (configuration or GeminiConfiguration()).model_copy(deep=True)
        self._client = genai.Client(
            api_key=api_key.get_secret_value(),
            vertexai=False,
            http_options=types.HttpOptions(
                api_version=self._configuration.api_version,
                timeout=max(1, int(self._configuration.timeout_seconds * 1000)),
                retry_options=types.HttpRetryOptions(attempts=1),
                httpx_client=http_client,
            ),
        )

    @property
    def configuration(self) -> GeminiConfiguration:
        return self._configuration.model_copy(deep=True)

    def close(self) -> None:
        self._client.close()

    def _generation_config(self, request: LLMGenerationRequest) -> dict[str, Any]:
        if request.max_output_tokens > self._configuration.max_output_tokens:
            raise PromptBudgetExceeded("Requested output exceeds Gemini's output limit.")
        return {
            "temperature": self._configuration.temperature,
            "candidateCount": 1,
            "maxOutputTokens": request.max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": request.response_schema,
            "thinkingConfig": {"thinking_level": self._configuration.thinking_level.upper()},
        }

    def count_tokens(self, request: LLMGenerationRequest) -> int:
        # SDK 2.22.0 rejects CountTokensConfig.system_instruction/generation_config
        # in Developer API mode. The public extra_body escape hatch supplies the
        # documented REST generateContentRequest, including the exact schema.
        body = {
            "model": f"models/{self._configuration.model}",
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "systemInstruction": {"role": "user", "parts": [{"text": request.system_prompt}]},
            "generationConfig": self._generation_config(request),
        }
        try:
            response = self._client.models.count_tokens(
                model=self._configuration.model,
                contents=request.user_prompt,
                config=types.CountTokensConfig(
                    http_options=types.HttpOptions(
                        extra_body={"contents": [], "generateContentRequest": body}
                    )
                ),
            )
        except (errors.APIError, httpx.HTTPError) as exc:
            raise GenerationProviderError("Gemini token counting failed.") from exc
        if response.total_tokens is None or response.total_tokens < 0:
            raise GenerationProviderError("Gemini returned no valid input token count.")
        return response.total_tokens

    def generate_structured(self, request: LLMGenerationRequest) -> LLMGenerationResponse:
        config = types.GenerateContentConfig.model_validate(self._generation_config(request))
        config.system_instruction = request.system_prompt
        config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        try:
            response = self._client.models.generate_content(
                model=self._configuration.model, contents=request.user_prompt, config=config
            )
        except (errors.APIError, httpx.HTTPError) as exc:
            raise GenerationProviderError("Gemini generation failed.") from exc
        except (ValidationError, json.JSONDecodeError) as exc:
            raise MalformedGenerationResponse(
                "Gemini returned a malformed response envelope."
            ) from exc
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise GenerationBlockedError("Gemini blocked the prompt.")
        if not response.candidates or len(response.candidates) != 1:
            raise MalformedGenerationResponse("Gemini returned no single answer candidate.")
        candidate = response.candidates[0]
        if candidate.finish_reason == types.FinishReason.MAX_TOKENS:
            raise IncompleteGenerationError("Gemini exhausted the output token budget.")
        if candidate.finish_reason in {
            types.FinishReason.SAFETY,
            types.FinishReason.RECITATION,
            types.FinishReason.BLOCKLIST,
            types.FinishReason.PROHIBITED_CONTENT,
            types.FinishReason.SPII,
        }:
            raise GenerationBlockedError("Gemini blocked the answer.")
        if candidate.finish_reason != types.FinishReason.STOP:
            raise IncompleteGenerationError("Gemini did not finish normally.")
        parts = candidate.content.parts if candidate.content else None
        answer_text = "".join(part.text or "" for part in (parts or []) if not part.thought)
        try:
            payload = json.loads(answer_text)
        except (ValueError, TypeError) as exc:
            raise MalformedGenerationResponse("Gemini returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise MalformedGenerationResponse("Gemini returned a non-object JSON answer.")
        usage = response.usage_metadata
        return LLMGenerationResponse(
            payload=payload,
            prompt_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            thinking_tokens=usage.thoughts_token_count if usage else None,
            request_id=response.response_id,
            model_version=response.model_version,
        )
