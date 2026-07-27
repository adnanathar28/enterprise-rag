from pathlib import Path

from pydantic import BaseModel, Field


class StoredSourceFile(BaseModel):
    original_filename: str
    stored_filename: str
    stored_path: Path
    size_bytes: int = Field(ge=0)
    extension: str
