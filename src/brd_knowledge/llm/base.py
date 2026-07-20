from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
