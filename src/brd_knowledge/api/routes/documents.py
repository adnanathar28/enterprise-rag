from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def list_documents() -> list[dict[str, str]]:
    return []
