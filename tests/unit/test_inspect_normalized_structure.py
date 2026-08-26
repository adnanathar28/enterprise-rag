import importlib.util
from pathlib import Path
from types import ModuleType

from brd_knowledge.schemas.document import DocumentMetadata, Page, ParsedDocument, TextBlock
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable, TableCell, TableRow


def load_inspector_script() -> ModuleType:
    script_path = Path("scripts/inspect_normalized_structure.py")
    spec = importlib.util.spec_from_file_location("inspect_normalized_structure", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_document() -> ParsedDocument:
    heading = TextBlock(
        block_id="block-heading",
        page_number=1,
        text="Master Data Setup",
        block_type="section_header",
        reading_order_index=10,
        source=SourceReference(document_id="doc-1", page_number=1),
    )
    paragraph = TextBlock(
        block_id="block-paragraph",
        page_number=1,
        text="This section defines the master data requirements.",
        block_type="paragraph",
        reading_order_index=20,
        source=SourceReference(document_id="doc-1", page_number=1),
    )
    cells = [
        TableCell(row_index=0, column_index=0, text="Field", is_header=True),
        TableCell(row_index=0, column_index=1, text="Required", is_header=True),
    ]
    table = ParsedTable(
        table_id="table-1",
        page_number=1,
        reading_order_index=15,
        rows=[TableRow(row_index=0, cells=cells)],
        cells=cells,
        source=SourceReference(document_id="doc-1", page_number=1),
    )
    section = DocumentSection(
        section_id="section-1",
        title="Master Data Setup",
        level=2,
        page_start=1,
        page_end=1,
        heading_block=heading,
        blocks=[heading, paragraph],
        paragraphs=[heading, paragraph],
    )
    return ParsedDocument(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="sample.pdf",
            file_type="pdf",
            page_count=1,
        ),
        pages=[
            Page(
                page_number=1,
                parse_status="success",
                blocks=[paragraph, heading],
                tables=[table],
            )
        ],
        sections=[section],
        blocks=[heading, paragraph],
        paragraphs=[heading, paragraph],
        tables=[table],
    )


def test_items_print_in_reading_order_with_section_membership_and_table() -> None:
    inspector = load_inspector_script()

    output = inspector.render_normalized_structure(synthetic_document())

    assert output.index("[10] section_header") < output.index("[15] table")
    assert output.index("[15] table") < output.index("[20] paragraph")
    assert "block_id=block-paragraph\n     section_id=section-1" in output
    assert "table_id=table-1" in output
    assert 'first_row="Field | Required"' in output


def test_health_flags_missing_table_section_and_section_summary() -> None:
    inspector = load_inspector_script()

    output = inspector.render_normalized_structure(synthetic_document())

    assert "section_id=section-1" in output
    assert "     level=2" in output
    assert "     child_sections=0" in output
    assert "- tables with no section: table-1" in output
    assert "- child_sections is empty for every section" in output
