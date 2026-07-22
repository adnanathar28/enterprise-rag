from types import SimpleNamespace

from brd_knowledge.parsing.docling_parser import DoclingDocumentParser
from brd_knowledge.schemas.document import ParserDiagnostic


def bbox() -> SimpleNamespace:
    return SimpleNamespace(l=10, t=20, r=110, b=70, coord_origin="TOPLEFT")


def prov(page_no: int = 1) -> SimpleNamespace:
    return SimpleNamespace(page_no=page_no, bbox=bbox())


def text_item(
    self_ref: str,
    label: str,
    text: str,
    page_no: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        self_ref=self_ref,
        label=label,
        text=text,
        prov=[prov(page_no)],
    )


def table_cell(
    text: str,
    row: int,
    column: int,
    column_header: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=column,
        end_col_offset_idx=column + 1,
        row_span=1,
        col_span=1,
        column_header=column_header,
        row_header=False,
        bbox=bbox(),
    )


def test_docling_adapter_normalizes_blocks_tables_images_and_pages() -> None:
    parser = DoclingDocumentParser(page_range=(1, 1))
    heading = text_item("#/texts/0", "section_header", "Scope", page_no=1)
    paragraph = text_item("#/texts/1", "text", "The system must track invoices.", page_no=1)
    raw_cells = [
        table_cell("Requirement", 0, 0, column_header=True),
        table_cell("Status", 0, 1, column_header=True),
        table_cell("Track invoices", 1, 0),
        table_cell("Required", 1, 1),
    ]
    table = SimpleNamespace(
        self_ref="#/tables/0",
        label="table",
        prov=[prov(1)],
        captions=[],
        data=SimpleNamespace(grid=[raw_cells[:2], raw_cells[2:]], table_cells=raw_cells),
    )
    picture = SimpleNamespace(self_ref="#/pictures/0", label="picture", prov=[prov(1)])
    docling_document = SimpleNamespace(
        pages={1: SimpleNamespace(size=SimpleNamespace(width=595.5, height=842.25))},
        texts=[heading, paragraph],
        tables=[table],
        pictures=[picture],
        iterate_items=lambda with_groups: [
            (heading, 1),
            (paragraph, 1),
            (table, 1),
            (picture, 1),
        ],
    )

    reading_order = parser._build_reading_order(docling_document)
    blocks = parser._build_blocks(docling_document, "doc-001", reading_order)
    tables = parser._build_tables(docling_document, "doc-001", reading_order)
    images = parser._build_images(docling_document, "doc-001", reading_order)
    diagnostics: list[ParserDiagnostic] = []
    pages = parser._build_pages(docling_document, "success", diagnostics, blocks, tables, images)
    sections = parser._build_sections("doc-001", blocks)

    assert blocks[0].block_type == "section_header"
    assert blocks[0].source is not None
    assert blocks[0].source.parser_item_id == "#/texts/0"
    assert tables[0].rows[0].cells[0].is_header is True
    assert tables[0].rows[1].cells[0].text == "Track invoices"
    assert images[0].source is not None
    assert images[0].source.parser_item_id == "#/pictures/0"
    assert pages[0].width == 595.5
    assert pages[0].blocks == blocks
    assert pages[0].tables == tables
    assert sections[0].title == "Scope"


def test_docling_adapter_marks_pages_with_diagnostics_as_failed() -> None:
    parser = DoclingDocumentParser()
    docling_document = SimpleNamespace(
        pages={42: SimpleNamespace(size=SimpleNamespace(width=595.5, height=842.25))},
    )
    diagnostic = ParserDiagnostic(
        severity="error",
        message="pipeline terminated early",
        page_number=42,
        stage="StandardPdfPipeline",
    )

    pages = parser._build_pages(
        docling_document=docling_document,
        parse_status="partial_success",
        diagnostics=[diagnostic],
        blocks=[],
        tables=[],
        images=[],
    )

    assert pages[0].page_number == 42
    assert pages[0].parse_status == "failed"
    assert pages[0].diagnostics == [diagnostic]
