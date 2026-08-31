import importlib.util
from pathlib import Path
from types import ModuleType

from brd_knowledge.schemas.chunk import Chunk


def load_chunk_script() -> ModuleType:
    script_path = Path("scripts/chunk_document.py")
    spec = importlib.util.spec_from_file_location("chunk_document", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_inspection_renders_sources_quality_and_preview() -> None:
    script = load_chunk_script()
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="Requirements",
        section_path=["Scope", "Requirements"],
        content_type="table",
        text="Recovered table content",
        page_start=2,
        page_end=3,
        source_table_ids=["table-1"],
        quality_notes=["Use native fallback."],
    )

    output = script.render_chunks([chunk])

    assert "chunk_id=chunk-1" in output
    assert "section=Scope > Requirements" in output
    assert "content_type=table" in output
    assert "pages=2-3" in output
    assert "source_table_ids=table-1" in output
    assert "quality=Use native fallback." in output
    assert 'preview="Recovered table content"' in output
