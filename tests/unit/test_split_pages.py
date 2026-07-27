from pathlib import Path

from pytest import MonkeyPatch

from brd_knowledge.parsing import split_pages
from brd_knowledge.schemas.document import DocumentMetadata, Page, ParsedDocument, ParserMetadata


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


def test_split_processing_keeps_failed_pages_non_fatal(monkeypatch: MonkeyPatch) -> None:
    def fake_process_document(
        document_path: Path,
        page_range: tuple[int, int] | None,
    ) -> ParsedDocument:
        if page_range == (2, 2):
            raise RuntimeError("synthetic page failure")
        assert page_range is not None
        return parsed_page(page_range[0])

    monkeypatch.setattr(split_pages, "process_document", fake_process_document)

    result = split_pages.process_split_pages(Path("sample.pdf"), (1, 3))

    assert result.parser_metadata is not None
    assert result.parser_metadata.parse_status == "partial_success"
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert [page.parse_status for page in result.pages] == ["success", "failed", "success"]
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].page_number == 2
    assert result.diagnostics[0].message == "synthetic page failure"


def test_split_processing_records_page_timeout(monkeypatch: MonkeyPatch) -> None:
    def fake_process_document_with_timeout(
        document_path: Path,
        page_number: int,
        timeout_seconds: float | None,
    ) -> ParsedDocument:
        if page_number == 2:
            raise TimeoutError("Page 2 exceeded timeout of 1.0 seconds.")
        return parsed_page(page_number)

    monkeypatch.setattr(
        split_pages,
        "process_document_with_timeout",
        fake_process_document_with_timeout,
    )

    result = split_pages.process_split_pages(
        Path("sample.pdf"),
        (1, 2),
        page_timeout_seconds=1.0,
    )

    assert result.parser_metadata is not None
    assert result.parser_metadata.parse_status == "partial_success"
    assert [page.parse_status for page in result.pages] == ["success", "failed"]
    assert result.diagnostics[0].error_type == "TimeoutError"
    assert result.diagnostics[0].message == "Page 2 exceeded timeout of 1.0 seconds."
