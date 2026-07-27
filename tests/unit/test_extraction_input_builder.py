from brd_knowledge.extraction.input_builder import build_requirement_extraction_units
from brd_knowledge.extraction.markdown_renderer import render_extraction_markdown
from brd_knowledge.schemas.document import DocumentMetadata, ParsedDocument, TextBlock
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.table import ParsedTable, TableCell, TableRow


def parsed_document() -> ParsedDocument:
    heading = TextBlock(
        block_id="page_1_texts-0",
        page_number=1,
        text="Invoice Validation",
        block_type="section_header",
        reading_order_index=0,
    )
    paragraph = TextBlock(
        block_id="page_1_texts-1",
        page_number=1,
        text="The system must validate invoice status before approval.",
        block_type="text",
        reading_order_index=1,
    )
    list_item = TextBlock(
        block_id="page_1_texts-2",
        page_number=1,
        text="Supervisor override is required for exceptions.",
        block_type="list_item",
        reading_order_index=2,
    )
    header_id = TableCell(
        cell_id="page_1_tables-0-r0-c0",
        row_index=0,
        column_index=0,
        text="ID",
        is_header=True,
    )
    header_description = TableCell(
        cell_id="page_1_tables-0-r0-c1",
        row_index=0,
        column_index=1,
        text="Description",
        is_header=True,
    )
    value_id = TableCell(
        cell_id="page_1_tables-0-r1-c0",
        row_index=1,
        column_index=0,
        text="BR-001",
    )
    value_description = TableCell(
        cell_id="page_1_tables-0-r1-c1",
        row_index=1,
        column_index=1,
        text="The system shall reject duplicate invoices.",
    )
    table = ParsedTable(
        table_id="page_1_tables-0",
        page_number=1,
        reading_order_index=3,
        rows=[
            TableRow(row_index=0, cells=[header_id, header_description]),
            TableRow(row_index=1, cells=[value_id, value_description]),
        ],
        cells=[header_id, header_description, value_id, value_description],
    )
    section = DocumentSection(
        section_id="section-1",
        title="Invoice Validation",
        level=1,
        page_start=1,
        heading_block=heading,
        blocks=[heading, paragraph, list_item],
        paragraphs=[heading, paragraph, list_item],
    )
    return ParsedDocument(
        metadata=DocumentMetadata(
            document_id="doc-001",
            filename="sample.pdf",
            file_type="pdf",
            page_count=1,
        ),
        sections=[section],
        blocks=[heading, paragraph, list_item],
        paragraphs=[heading, paragraph, list_item],
        tables=[table],
    )


def test_build_requirement_extraction_units_from_blocks_and_table_rows() -> None:
    units = build_requirement_extraction_units(parsed_document())

    assert [unit.content_type for unit in units] == [
        "section_header",
        "paragraph",
        "list_item",
        "table_row",
    ]
    assert units[1].section_id == "section-1"
    assert units[1].section_title == "Invoice Validation"
    assert units[1].source_refs[0].block_id == "page_1_texts-1"
    assert units[3].text == (
        "ID: BR-001 | Description: The system shall reject duplicate invoices."
    )
    assert units[3].source_refs[0].table_id == "page_1_tables-0"
    assert units[3].source_refs[0].cell_ids == [
        "page_1_tables-0-r1-c0",
        "page_1_tables-0-r1-c1",
    ]


def test_render_extraction_markdown_includes_provenance_markers() -> None:
    markdown = render_extraction_markdown(build_requirement_extraction_units(parsed_document()))

    assert "<!-- extraction_view_version: 1 -->" in markdown
    assert "<!-- page: 1 -->" in markdown
    assert "<!-- section_id: section-1 | page: 1 -->" in markdown
    assert "## Invoice Validation" in markdown
    assert (
        "<!-- unit_id: unit-page_1_texts-1 | type: paragraph | page: 1 | "
        "section_id: section-1 | block_id: page_1_texts-1 -->"
    ) in markdown
    assert "- Supervisor override is required for exceptions." in markdown
    assert (
        "table_id: page_1_tables-0 | "
        "cell_ids: page_1_tables-0-r1-c0,page_1_tables-0-r1-c1"
    ) in markdown
