from brd_knowledge.llm.base import (
    LLMConfiguration,
    LLMGenerationRequest,
    LLMGenerationResponse,
    LLMProvider,
)


class NoopLLMProvider(LLMProvider):
    @property
    def configuration(self) -> LLMConfiguration:
        return LLMConfiguration(
            provider="noop",
            model="noop",
            context_window_tokens=1,
        )

    def count_tokens(self, system_prompt: str, user_prompt: str) -> int:
        raise NotImplementedError("NoopLLMProvider cannot count tokens.")

    def generate_structured(self, request: LLMGenerationRequest) -> LLMGenerationResponse:
        raise NotImplementedError("NoopLLMProvider cannot generate answers.")
