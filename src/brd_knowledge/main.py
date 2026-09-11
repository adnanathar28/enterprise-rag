from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from brd_knowledge.api.dependencies import reranker_scorer_dependency
from brd_knowledge.api.routes.capabilities import router as capabilities_router
from brd_knowledge.api.routes.documents import router as documents_router
from brd_knowledge.api.routes.requirements import router as requirements_router
from brd_knowledge.core.config import get_settings
from brd_knowledge.core.exceptions import RetrievalRerankingError

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        reranker_scorer_dependency().load()
    except Exception as exc:
        raise RetrievalRerankingError(
            "Pinned retrieval reranker failed to initialize during application startup."
        ) from exc
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(capabilities_router, prefix="/capabilities", tags=["capabilities"])
app.include_router(requirements_router, prefix="/requirements", tags=["requirements"])
