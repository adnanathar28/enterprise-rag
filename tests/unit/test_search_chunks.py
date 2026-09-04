import importlib.util
from pathlib import Path
from types import ModuleType

from brd_knowledge.schemas.retrieval import RetrievedChunk


def load_search_script() -> ModuleType:
    script_path = Path("scripts/search_chunks.py")
    spec = importlib.util.spec_from_file_location("search_chunks", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_results_shows_score_section_sources_and_preview() -> None:
    result = RetrievedChunk(
        rank=1,
        cosine_distance=0.2,
        similarity=0.8,
        chunk_id="chunk-1",
        document_id="doc-1",
        section_path=["Scope", "Audit"],
        content_type="table",
        text="Audit field | Required value",
        page_start=4,
        page_end=5,
        source_table_ids=["table-1"],
    )

    output = load_search_script().render_results([result])

    assert "[1] similarity=0.800000" in output
    assert "section=Scope > Audit" in output
    assert "pages=4-5" in output
    assert "source_table_ids=table-1" in output
    assert 'preview="Audit field | Required value"' in output
