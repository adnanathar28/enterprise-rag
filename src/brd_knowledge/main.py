from fastapi import FastAPI

from brd_knowledge.api.routes.documents import router as documents_router
from brd_knowledge.api.routes.requirements import router as requirements_router
from brd_knowledge.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(requirements_router, prefix="/requirements", tags=["requirements"])
