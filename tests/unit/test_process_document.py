import importlib.util
from pathlib import Path
from types import ModuleType

from brd_knowledge.schemas.document import (
    DocumentMetadata,
    Page,
    ParsedDocument,
    ParserMetadata,
)


def load_process_document_script() -> ModuleType:
    script_path = Path("scripts/process_document.py")
    spec = importlib.util.spec_from_file_location("process_document_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parsed_page(page_number: int) -> ParsedDocument:
    return ParsedDocument(
        metadata=DocumentMetadata(
            document_id="sample",
            filename="sample.pdf",
            file_type="pdf",
            page_count=1,
            parser_name="docling",
        ),
        parser_metadata=ParserMetadata(
            parser_name="docling",
            parse_status="success",
            parse_strategy="page_range",
        ),
        pages=[
            Page(
                page_number=page_number,
                parse_status="success",
                parser_name="docling",
            )
        ],
    )


def test_split_processing_keeps_failed_pages_non_fatal(monkeypatch) -> None:
    process_document_script = load_process_document_script()

    def fake_process_document(
        document_path: Path,
        page_range: tuple[int, int] | None,
    ) -> ParsedDocument:
        if page_range == (2, 2):
            raise RuntimeError("synthetic page failure")
        assert page_range is not None
        return parsed_page(page_range[0])

    monkeypatch.setattr(process_document_script, "process_document", fake_process_document)

    result = process_document_script.process_split_pages(Path("sample.pdf"), (1, 3))

    assert result.parser_metadata is not None
    assert result.parser_metadata.parse_status == "partial_success"
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert [page.parse_status for page in result.pages] == ["success", "failed", "success"]
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].page_number == 2
    assert result.diagnostics[0].message == "synthetic page failure"
