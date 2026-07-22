import pytest
from pydantic import ValidationError

from brd_knowledge.schemas import (
    BoundingBox,
    DocumentBlock,
    DocumentMetadata,
    ExtractedRequirement,
    Page,
    ParsedDocument,
    ParsedImage,
    ParsedTable,
    ParserDiagnostic,
    ParserMetadata,
    SourceReference,
    TableCell,
    TableRow,
    TextBlock,
)
from brd_knowledge.schemas.section import DocumentSection


def test_parsed_document_schema_validation() -> None:
    metadata = DocumentMetadata(
        document_id="doc-001",
        filename="sample.pdf",
        file_type="pdf",
        page_count=2,
    )
    bounding_box = BoundingBox(
        page_number=1,
        x0=0,
        y0=0,
        x1=100,
        y1=50,
        coordinate_origin="top_left",
    )
    source = SourceReference(
        document_id="doc-001",
        page_number=1,
        block_id="block-001",
        reading_order_index=0,
        text_excerpt="The system shall support approval workflows.",
        bounding_box=bounding_box,
    )
    paragraph = TextBlock(
        block_id="block-001",
        page_number=1,
        text="The system shall support approval workflows.",
        reading_order_index=0,
        bounding_box=bounding_box,
        source=source,
    )
    row = TableRow(
        row_index=0,
        cells=[
            TableCell(
                cell_id="cell-001",
                row_index=0,
                column_index=0,
                text="Requirement",
                bounding_box=bounding_box,
            )
        ],
    )
    table = ParsedTable(
        table_id="table-001",
        page_number=1,
        page_numbers=[1],
        reading_order_index=1,
        rows=[row],
        cells=row.cells,
        bounding_box=bounding_box,
    )
    image = ParsedImage(
        image_id="image-001",
        page_number=1,
        reading_order_index=2,
        bounding_box=bounding_box,
    )
    page = Page(
        page_number=1,
        width=595.5,
        height=842.25,
        parse_status="success",
        parser_name="docling",
        parser_version="2.114.0",
        blocks=[paragraph],
        tables=[table],
        images=[image],
    )
    section = DocumentSection(
        section_id="section-001",
        title="Workflow Requirements",
        level=1,
        page_start=1,
        source=source,
        blocks=[paragraph],
        paragraphs=[paragraph],
        tables=[table],
    )
    parser_metadata = ParserMetadata(
        parser_name="docling",
        parser_version="2.114.0",
        parse_status="success",
        parse_strategy="page_range",
    )

    parsed_document = ParsedDocument(
        metadata=metadata,
        parser_metadata=parser_metadata,
        pages=[page],
        sections=[section],
        blocks=[paragraph],
        paragraphs=[paragraph],
        tables=[table],
        images=[image],
    )

    assert parsed_document.metadata.document_id == "doc-001"
    assert parsed_document.pages[0].page_number == 1
    assert parsed_document.pages[0].parse_status == "success"
    assert parsed_document.sections[0].paragraphs[0].page_number == 1
    assert parsed_document.tables[0].rows[0].cells[0].text == "Requirement"
    assert parsed_document.images[0].image_id == "image-001"


def test_requirement_source_reference_validation() -> None:
    source = SourceReference(
        document_id="doc-001",
        page_number=1,
        section_id="section-001",
        text_excerpt="The system shall support approval workflows.",
    )
    requirement = ExtractedRequirement(
        requirement_id="REQ-001",
        document_id="doc-001",
        text="The system shall support approval workflows.",
        source=source,
        confidence=0.9,
    )

    assert requirement.source.page_number == 1
    assert requirement.confidence == 0.9


def test_page_numbers_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(page_number=0, x0=0, y0=0, x1=100, y1=50)


def test_parser_diagnostics_can_represent_partial_page_failure() -> None:
    diagnostic = ParserDiagnostic(
        severity="error",
        parser_name="docling",
        parser_version="2.114.0",
        stage="preprocess",
        page_number=24,
        error_type="std::bad_alloc",
        message="Stage preprocess failed for page 24.",
    )
    page = Page(
        page_number=24,
        parse_status="failed",
        parser_name="docling",
        diagnostics=[diagnostic],
        quality_notes=["Source PDF table has overlapping text."],
    )

    assert page.parse_status == "failed"
    assert page.diagnostics[0].stage == "preprocess"
    assert page.quality_notes == ["Source PDF table has overlapping text."]


def test_blocks_preserve_reading_order_and_source_reference() -> None:
    source = SourceReference(
        document_id="doc-001",
        page_number=2,
        parser_item_id="#/texts/3",
        reading_order_index=3,
    )
    block = DocumentBlock(
        block_id="block-003",
        page_number=2,
        text="Customer delivery is validated at invoice level.",
        block_type="list_item",
        reading_order_index=3,
        source=source,
    )

    assert block.reading_order_index == 3
    assert block.source is not None
    assert block.source.parser_item_id == "#/texts/3"
