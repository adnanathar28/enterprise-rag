from pathlib import Path

from pytest import MonkeyPatch

from brd_knowledge.parsing import split_pages
from brd_knowledge.schemas.document import (
    DocumentMetadata,
    Page,
    ParsedDocument,
    ParserMetadata,
    TextBlock,
)
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable


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


def test_merge_removes_repeated_headers_and_keeps_continuation_table_in_section() -> None:
    heading = TextBlock(
        block_id="heading",
        page_number=1,
        reading_order_index=1,
        block_type="section_header",
        text="1) Requirements",
        source=SourceReference(document_id="sample", page_number=1, block_id="heading"),
    )
    first_header = TextBlock(
        block_id="header",
        page_number=1,
        reading_order_index=0,
        block_type="page_header",
        text="Sample Document",
        source=SourceReference(document_id="sample", page_number=1, block_id="header"),
    )
    second_header = TextBlock(
        block_id="header",
        page_number=2,
        reading_order_index=0,
        block_type="section_header",
        text="Sample Document",
        source=SourceReference(document_id="sample", page_number=2, block_id="header"),
    )
    continuation_table = ParsedTable(
        table_id="table",
        page_number=2,
        page_numbers=[2],
        reading_order_index=1,
        source=SourceReference(document_id="sample", page_number=2, table_id="table"),
    )
    first_page = parsed_page(1)
    first_page.blocks = [first_header, heading]
    first_page.paragraphs = [first_header, heading]
    first_page.pages[0].blocks = [first_header, heading]
    second_page = parsed_page(2)
    second_page.blocks = [second_header]
    second_page.paragraphs = [second_header]
    second_page.tables = [continuation_table]
    second_page.pages[0].blocks = [second_header]
    second_page.pages[0].tables = [continuation_table]

    merged = split_pages.merge_parsed_documents(
        [
            split_pages.prefix_document_ids_for_split_page(first_page, 1),
            split_pages.prefix_document_ids_for_split_page(second_page, 2),
        ]
    )

    assert [item.text for item in merged.blocks] == ["1) Requirements"]
    assert [section.title for section in merged.sections] == ["1) Requirements"]
    assert [table.table_id for table in merged.sections[0].tables] == ["page_2_table"]
