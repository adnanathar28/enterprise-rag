from brd_knowledge.llm.base import LLMProvider


class NoopLLMProvider(LLMProvider):
    def name(self) -> str:
        return "noop"
