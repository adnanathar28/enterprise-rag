from brd_knowledge.core.config import Settings
from brd_knowledge.core.exceptions import GenerationProviderError
from brd_knowledge.llm.base import LLMProvider
from brd_knowledge.llm.gemini import GeminiConfiguration, GeminiLLMProvider
from brd_knowledge.llm.local_qwen import LocalQwenConfiguration, LocalQwenProvider


def create_llm_provider(
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    selected = provider or settings.llm_provider
    if selected == "local_qwen":
        return LocalQwenProvider(
            LocalQwenConfiguration.model_validate(
                {
                    "model": model or settings.local_qwen_model,
                    "base_url": settings.local_qwen_base_url,
                    "context_window_tokens": settings.local_qwen_context_tokens,
                    "timeout_seconds": settings.local_qwen_timeout_seconds,
                }
            )
        )
    if selected != "gemini":
        raise ValueError("Unknown LLM provider.")
    if (
        settings.gemini_api_key is None
        or not settings.gemini_api_key.get_secret_value().strip()
        or settings.gemini_api_key.get_secret_value() == "replace-with-your-api-key"
    ):
        raise GenerationProviderError("Set GEMINI_API_KEY in the environment or ignored .env.")
    return GeminiLLMProvider(
        settings.gemini_api_key,
        GeminiConfiguration.model_validate(
            {
                "model": model or settings.gemini_model,
                "temperature": settings.gemini_temperature,
                "timeout_seconds": settings.gemini_timeout_seconds,
                "safety_margin_tokens": settings.gemini_safety_margin_tokens,
            }
        ),
    )
