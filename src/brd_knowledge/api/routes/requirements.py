from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def list_requirements() -> list[dict[str, str]]:
    return []
