from pathlib import Path

import pytest

from brd_knowledge.parsing.base import DocumentParser
from brd_knowledge.parsing.options import ParseOptions
from brd_knowledge.schemas.document import DocumentMetadata, Page, ParsedDocument, ParserMetadata
from brd_knowledge.schemas.table import ParsedTable, TableCell, TableRow
from brd_knowledge.services.ingestion_service import IngestionService


class FakeParser(DocumentParser):
    parser_name = "fake"

    def __init__(
        self,
        supported_suffix: str = ".pdf",
        result: ParsedDocument | None = None,
    ) -> None:
        self.supported_suffix = supported_suffix
        self.result = result
        self.parsed_paths: list[Path] = []

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix == self.supported_suffix

    def parse(self, file_path: Path) -> ParsedDocument:
        self.parsed_paths.append(file_path)
        if self.result is not None:
            return self.result
        return parsed_document(file_path)


def parsed_document(file_path: Path) -> ParsedDocument:
    return ParsedDocument(
        metadata=DocumentMetadata(
            document_id=file_path.stem,
            filename=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            page_count=0,
            parser_name="fake",
        )
    )


def test_ingest_uses_registered_parser_for_normal_parse() -> None:
    parser = FakeParser()
    service = IngestionService(parsers=[parser])
    file_path = Path("sample.pdf")

    result = service.ingest(file_path)

    assert result.metadata.filename == "sample.pdf"
    assert parser.parsed_paths == [file_path]


def test_ingest_routes_split_pages_through_split_processor() -> None:
    parser = FakeParser()
    calls: list[tuple[Path, tuple[int, int], float | None]] = []

    def fake_split_processor(
        document_path: Path,
        page_range: tuple[int, int],
        page_timeout_seconds: float | None = None,
    ) -> ParsedDocument:
        calls.append((document_path, page_range, page_timeout_seconds))
        return parsed_document(document_path)

    service = IngestionService(
        parsers=[parser],
        split_page_processor=fake_split_processor,
    )
    file_path = Path("sample.pdf")

    result = service.ingest(
        file_path,
        ParseOptions(
            page_range=(1, 3),
            split_pages=True,
            page_timeout_seconds=10,
        ),
    )

    assert result.metadata.filename == "sample.pdf"
    assert calls == [(file_path, (1, 3), 10)]
    assert parser.parsed_paths == []


def test_ingest_rejects_unsupported_file() -> None:
    service = IngestionService(parsers=[FakeParser()])

    with pytest.raises(ValueError, match="No parser registered"):
        service.ingest(Path("sample.xlsx"))


def test_ingest_with_result_summarizes_parsed_document() -> None:
    document = ParsedDocument(
        metadata=DocumentMetadata(
            document_id="sample",
            filename="sample.pdf",
            file_type="pdf",
            page_count=2,
            parser_name="fake",
        ),
        parser_metadata=ParserMetadata(
            parser_name="fake",
            parse_status="partial_success",
        ),
        pages=[
            Page(page_number=1, parse_status="success"),
            Page(page_number=2, parse_status="failed"),
        ],
        tables=[
            ParsedTable(
                table_id="table-1",
                rows=[
                    TableRow(
                        row_index=0,
                        cells=[TableCell(row_index=0, column_index=0, text="Requirement")],
                    )
                ],
                cells=[TableCell(row_index=0, column_index=0, text="Requirement")],
            )
        ],
    )

    parser = FakeParser(result=document)
    service = IngestionService(parsers=[parser])

    result = service.ingest_with_result(Path("sample.pdf"))

    assert result.document_id == "sample"
    assert result.parse_status == "partial_success"
    assert result.page_count == 2
    assert result.table_count == 1
    assert result.table_cell_count == 1
    assert result.failed_page_count == 1
    assert result.parsed_document == document


def test_parse_options_validate_split_page_range() -> None:
    with pytest.raises(ValueError, match="Split-page parsing requires a page range"):
        ParseOptions(split_pages=True)

    with pytest.raises(ValueError, match="start page 4 is after end page 2"):
        ParseOptions(page_range=(4, 2))

    with pytest.raises(ValueError, match="greater than 0"):
        ParseOptions(page_range=(1, 2), split_pages=True, page_timeout_seconds=0)
