import pytest
from pydantic import ValidationError

from brd_knowledge.schemas import (
    BoundingBox,
    DocumentMetadata,
    ExtractedRequirement,
    ParsedDocument,
    ParsedTable,
    SourceReference,
    TableCell,
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
    bounding_box = BoundingBox(page_number=1, x0=0, y0=0, x1=100, y1=50)
    paragraph = TextBlock(
        block_id="block-001",
        page_number=1,
        text="The system shall support approval workflows.",
        bounding_box=bounding_box,
    )
    table = ParsedTable(
        table_id="table-001",
        page_number=1,
        cells=[TableCell(row_index=0, column_index=0, text="Requirement")],
    )
    section = DocumentSection(
        section_id="section-001",
        title="Workflow Requirements",
        level=1,
        page_start=1,
        paragraphs=[paragraph],
        tables=[table],
    )

    parsed_document = ParsedDocument(
        metadata=metadata,
        pages=[1, 2],
        sections=[section],
        paragraphs=[paragraph],
        tables=[table],
    )

    assert parsed_document.metadata.document_id == "doc-001"
    assert parsed_document.sections[0].paragraphs[0].page_number == 1
    assert parsed_document.tables[0].cells[0].text == "Requirement"


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
