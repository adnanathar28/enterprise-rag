from typing import Annotated

from fastapi import APIRouter, Depends

from brd_knowledge.api.dependencies import settings_dependency
from brd_knowledge.core.config import Settings
from brd_knowledge.schemas.query import ApplicationCapabilities, ProviderCapability

router = APIRouter()


@router.get("", response_model=ApplicationCapabilities)
def get_capabilities(
    settings: Annotated[Settings, Depends(settings_dependency)],
) -> ApplicationCapabilities:
    gemini_configured = bool(
        settings.gemini_api_key
        and settings.gemini_api_key.get_secret_value().strip()
        and settings.gemini_api_key.get_secret_value() != "replace-with-your-api-key"
    )
    return ApplicationCapabilities(
        providers=[
            ProviderCapability(
                provider="gemini",
                label="Gemini",
                model=settings.gemini_model,
                configured=gemini_configured,
                is_default=settings.llm_provider == "gemini",
            ),
            ProviderCapability(
                provider="local_qwen",
                label="Local Qwen",
                model=settings.local_qwen_model,
                configured=True,
                is_default=settings.llm_provider == "local_qwen",
            ),
        ],
        allowed_document_extensions=sorted(settings.allowed_document_extensions),
        max_upload_size_bytes=settings.max_upload_size_bytes,
    )
