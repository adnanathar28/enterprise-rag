from pydantic import ValidationError

from brd_knowledge.core.exceptions import (
    GenerationProviderError,
    MalformedGenerationResponse,
    PromptBudgetExceeded,
)
from brd_knowledge.generation.citations import validate_and_resolve_citations
from brd_knowledge.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from brd_knowledge.llm.base import LLMGenerationRequest, LLMProvider
from brd_knowledge.schemas.generation import (
    GenerationMetadata,
    GroundedAnswer,
    GroundedAnswerRequest,
    ModelAnswerPayload,
)

INSUFFICIENT_ANSWER = "The supplied evidence is insufficient to answer this question."


class GroundedAnswerService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def generate(self, request: GroundedAnswerRequest) -> GroundedAnswer:
        if not request.context.evidence or not request.context.rendered_text.strip():
            return GroundedAnswer(
                answer_text=INSUFFICIENT_ANSWER,
                cited_evidence_ids=[],
                citations=[],
                insufficient_evidence=True,
                metadata=self._metadata(request.max_output_tokens),
            )

        user_prompt = build_user_prompt(request.question, request.context)
        prompt_tokens = self._count_tokens(SYSTEM_PROMPT, user_prompt)
        configuration = self._provider.configuration
        required_tokens = (
            prompt_tokens
            + request.max_output_tokens
            + configuration.safety_margin_tokens
        )
        if required_tokens > configuration.context_window_tokens:
            raise PromptBudgetExceeded(
                f"Prompt requires {required_tokens} tokens including output reserve and safety "
                f"margin; model limit is {configuration.context_window_tokens}."
            )

        provider_request = LLMGenerationRequest(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_output_tokens=request.max_output_tokens,
            response_schema=ModelAnswerPayload.model_json_schema(),
        )
        try:
            response = self._provider.generate_structured(provider_request)
        except Exception as exc:
            raise GenerationProviderError("The LLM provider call failed.") from exc

        try:
            payload = ModelAnswerPayload.model_validate(response.payload)
        except ValidationError as exc:
            raise MalformedGenerationResponse(
                "The provider response did not match the grounded answer schema."
            ) from exc

        cited_ids, citations = validate_and_resolve_citations(payload, request.context)
        return GroundedAnswer(
            answer_text=payload.answer_text,
            cited_evidence_ids=cited_ids,
            citations=citations,
            insufficient_evidence=payload.insufficient_evidence,
            metadata=self._metadata(
                request.max_output_tokens,
                prompt_tokens=(
                    response.prompt_tokens
                    if response.prompt_tokens is not None
                    else prompt_tokens
                ),
                output_tokens=response.output_tokens,
                provider_request_id=response.request_id,
            ),
        )

    def _count_tokens(self, system_prompt: str, user_prompt: str) -> int:
        try:
            count = self._provider.count_tokens(system_prompt, user_prompt)
        except Exception as exc:
            raise GenerationProviderError(
                "The LLM provider could not count prompt tokens."
            ) from exc
        if count < 0:
            raise GenerationProviderError("The LLM provider returned a negative token count.")
        return count

    def _metadata(
        self,
        max_output_tokens: int,
        *,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        provider_request_id: str | None = None,
    ) -> GenerationMetadata:
        configuration = self._provider.configuration
        return GenerationMetadata(
            provider=configuration.provider,
            model=configuration.model,
            revision=configuration.revision,
            max_output_tokens=max_output_tokens,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            provider_request_id=provider_request_id,
        )
