import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError


def load_embed_script() -> ModuleType:
    script_path = Path("scripts/embed_chunks.py")
    spec = importlib.util.spec_from_file_location("embed_chunks", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_chunks_validates_existing_chunk_schema(tmp_path: Path) -> None:
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "content_type": "prose",
                    "text": "A requirement",
                    "page_start": 1,
                    "page_end": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    chunks = load_embed_script().load_chunks(path)

    assert chunks[0].chunk_id == "chunk-1"


def test_load_chunks_rejects_invalid_chunk(tmp_path: Path) -> None:
    path = tmp_path / "chunks.json"
    path.write_text("[{}]", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_embed_script().load_chunks(path)
