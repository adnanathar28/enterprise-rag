from pathlib import Path
from types import SimpleNamespace

import pymupdf

from brd_knowledge.parsing.docling_parser import DoclingDocumentParser, _NativeWord
from brd_knowledge.schemas.document import DocumentBlock, ParserDiagnostic
from brd_knowledge.schemas.source import BoundingBox, SourceReference
from brd_knowledge.schemas.table import ParsedTable, TableCell, TableRow


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
    row_span: int = 1,
    column_span: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=column,
        end_col_offset_idx=column + column_span,
        row_span=row_span,
        col_span=column_span,
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
    document_blocks: list[DocumentBlock] = list(blocks)
    pages = parser._build_pages(
        docling_document,
        "success",
        diagnostics,
        document_blocks,
        tables,
        images,
    )
    sections = parser._build_sections("doc-001", blocks, tables)

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
    assert sections[0].tables == tables


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


def test_docling_adapter_excludes_page_footers_from_semantic_blocks() -> None:
    parser = DoclingDocumentParser(page_range=(1, 1))
    heading = text_item("#/texts/0", "section_header", "1) Scope")
    footer = text_item("#/texts/1", "page_footer", "12")
    docling_document = SimpleNamespace(texts=[heading, footer])

    blocks = parser._build_blocks(
        docling_document,
        document_id="doc-001",
        reading_order={"#/texts/0": 1},
    )

    assert [block.block_id for block in blocks] == ["texts-0"]


def test_docling_adapter_uses_document_section_reconstruction() -> None:
    parser = DoclingDocumentParser()
    parent = text_item("#/texts/0", "section_header", "1) Master Data", page_no=1)
    child = text_item("#/texts/1", "section_header", "1.1 Vehicles", page_no=1)
    paragraph = text_item("#/texts/2", "text", "Vehicle details.", page_no=1)
    docling_document = SimpleNamespace(texts=[paragraph, child, parent])
    reading_order = {
        "#/texts/0": 1,
        "#/texts/1": 2,
        "#/texts/2": 3,
    }

    blocks = parser._build_blocks(docling_document, "doc-001", reading_order)
    sections = parser._build_sections("doc-001", blocks)

    assert [section.title for section in sections] == ["1) Master Data"]
    assert [section.title for section in sections[0].child_sections] == ["1.1 Vehicles"]
    assert [block.block_id for block in sections[0].child_sections[0].blocks] == [
        "texts-1",
        "texts-2",
    ]


def test_docling_adapter_uses_unique_logical_table_cells_for_rows() -> None:
    parser = DoclingDocumentParser(page_range=(1, 1))
    spanned_header = table_cell(
        "Length Width Height",
        0,
        2,
        column_header=True,
        column_span=3,
    )
    raw_cells = [
        table_cell("Description", 0, 0, column_header=True),
        spanned_header,
        table_cell("Small", 1, 0),
        table_cell("20", 1, 2),
        table_cell("15", 1, 3),
        table_cell("12", 1, 4),
    ]
    table = SimpleNamespace(
        self_ref="#/tables/0",
        label="table",
        prov=[prov(1)],
        captions=[],
        data=SimpleNamespace(
            grid=[
                [raw_cells[0], spanned_header, spanned_header, spanned_header],
                raw_cells[2:],
            ],
            table_cells=raw_cells,
        ),
    )
    docling_document = SimpleNamespace(
        tables=[table],
        iterate_items=lambda with_groups: [(table, 1)],
    )

    tables = parser._build_tables(
        docling_document,
        document_id="doc-001",
        reading_order=parser._build_reading_order(docling_document),
    )

    assert len(tables[0].rows[0].cells) == 2
    assert tables[0].rows[0].cells[1].text == "Length Width Height"
    assert tables[0].rows[0].cells[1].column_span == 3
    assert len(tables[0].cells) == 6


def test_docling_adapter_adds_native_table_text_and_flags_low_coverage(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "table.pdf"
    document = pymupdf.open()
    page = document.new_page(width=300, height=300)
    page.insert_text((50, 80), "Known value omitted requirement")
    document.save(pdf_path)
    document.close()

    table = ParsedTable(
        table_id="table-1",
        page_number=1,
        page_numbers=[1],
        bounding_box=BoundingBox(
            page_number=1,
            x0=40,
            y0=240,
            x1=260,
            y1=200,
            coordinate_origin="bottom_left",
        ),
        cells=[TableCell(row_index=0, column_index=0, text="Known value")],
    )

    DoclingDocumentParser()._attach_native_table_evidence(pdf_path, [table])

    assert table.native_text == "Known value omitted requirement"
    assert table.native_text_coverage == 0.5
    assert table.quality_notes == [
        "Structured table text covers only 50.0% of native table-region tokens; "
        "use native_text as a fallback."
    ]


def test_docling_adapter_keeps_high_coverage_table_unflagged(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    document = pymupdf.open()
    page = document.new_page(width=300, height=300)
    page.insert_text((50, 80), "Complete table text")
    document.save(pdf_path)
    document.close()

    table = ParsedTable(
        table_id="table-1",
        page_number=1,
        page_numbers=[1],
        bounding_box=BoundingBox(
            page_number=1,
            x0=40,
            y0=240,
            x1=260,
            y1=200,
            coordinate_origin="bottom_left",
        ),
        cells=[TableCell(row_index=0, column_index=0, text="Complete table text")],
    )

    DoclingDocumentParser()._attach_native_table_evidence(pdf_path, [table])

    assert table.native_text == "Complete table text"
    assert table.native_text_coverage == 1.0
    assert table.quality_notes == []


def test_docling_coordinate_origin_recognizes_docling_enum_strings() -> None:
    parser = DoclingDocumentParser()

    assert parser._coordinate_origin("CoordOrigin.TOPLEFT") == "top_left"
    assert parser._coordinate_origin("CoordOrigin.BOTTOMLEFT") == "bottom_left"


def native_box(x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(
        page_number=1,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        coordinate_origin="top_left",
    )


def native_word(text: str, x0: float, y0: float, x1: float, y1: float) -> _NativeWord:
    return _NativeWord(x0=x0, y0=y0, x1=x1, y1=y1, text=text)


def malformed_two_row_group() -> tuple[ParsedTable, list[_NativeWord]]:
    cells = [
        TableCell(
            cell_id="table-1-r1-c0",
            row_index=1,
            column_index=0,
            text="Model A",
            bounding_box=native_box(10, 14, 70, 24),
        ),
        TableCell(
            cell_id="table-1-r1-c1",
            row_index=1,
            column_index=1,
            text="GraphRAG VersionRAG",
            bounding_box=native_box(100, 10, 170, 28),
            source=SourceReference(
                document_id="doc-1",
                page_number=1,
                table_id="table-1",
                cell_id="table-1-r1-c1",
                text_excerpt="GraphRAG VersionRAG",
                bounding_box=native_box(100, 10, 170, 28),
            ),
        ),
        TableCell(
            cell_id="table-1-r1-c2",
            row_index=1,
            column_index=2,
            text="100",
            bounding_box=native_box(200, 10, 230, 18),
        ),
        TableCell(
            cell_id="table-1-r1-c3",
            row_index=1,
            column_index=3,
            text="10min",
            bounding_box=native_box(300, 10, 340, 18),
        ),
        TableCell(
            cell_id="table-1-r2-c2",
            row_index=2,
            column_index=2,
            text="50",
            bounding_box=native_box(200, 20, 230, 28),
        ),
        TableCell(
            cell_id="table-1-r2-c3",
            row_index=2,
            column_index=3,
            text="5min",
            bounding_box=native_box(300, 20, 340, 28),
        ),
    ]
    table = ParsedTable(
        table_id="table-1",
        page_number=1,
        page_numbers=[1],
        rows=[
            TableRow(row_index=1, cells=cells[:4]),
            TableRow(row_index=2, cells=cells[4:]),
        ],
        cells=cells,
    )
    words = [
        native_word("Model", 10, 14, 42, 24),
        native_word("A", 45, 14, 52, 24),
        native_word("GraphRAG", 100, 10, 140, 18),
        native_word("VersionRAG", 100, 20, 150, 28),
        native_word("100", 200, 10, 230, 18),
        native_word("10min", 300, 10, 340, 18),
        native_word("50", 200, 20, 230, 28),
        native_word("5min", 300, 20, 340, 28),
    ]
    return table, words


def test_native_geometry_repairs_malformed_two_row_group() -> None:
    table, words = malformed_two_row_group()

    repaired = DoclingDocumentParser()._repair_table_from_native_geometry(table, words)

    assert repaired is True
    rendered_cells = [
        (cell.row_index, cell.column_index, cell.text, cell.row_span) for cell in table.cells
    ]
    assert rendered_cells == [
        (1, 0, "Model A", 2),
        (1, 1, "GraphRAG", 1),
        (1, 2, "100", 1),
        (1, 3, "10min", 1),
        (2, 1, "VersionRAG", 1),
        (2, 2, "50", 1),
        (2, 3, "5min", 1),
    ]
    split_cells = [cell for cell in table.cells if cell.column_index == 1]
    assert [cell.source.cell_id for cell in split_cells if cell.source is not None] == [
        "table-1-r1-c1",
        "table-1-r2-c1",
    ]
    assert all(cell.source is not None for cell in split_cells)


def test_native_geometry_leaves_normal_table_unchanged() -> None:
    table, words = malformed_two_row_group()
    approach = table.cells[1]
    approach.text = "GraphRAG"
    approach.bounding_box = native_box(100, 10, 140, 18)
    table.cells.insert(
        4,
        TableCell(
            cell_id="table-1-r2-c1",
            row_index=2,
            column_index=1,
            text="VersionRAG",
            bounding_box=native_box(100, 20, 150, 28),
        ),
    )
    table.rows[1].cells.insert(0, table.cells[4])
    before = table.model_dump(mode="json")

    repaired = DoclingDocumentParser()._repair_table_from_native_geometry(table, words)

    assert repaired is False
    assert table.model_dump(mode="json") == before


def test_native_geometry_leaves_existing_row_span_unchanged() -> None:
    table, words = malformed_two_row_group()
    table.cells[0].row_span = 2
    table.rows[0].cells[0].row_span = 2
    table.cells[1].text = "GraphRAG"
    table.cells[1].bounding_box = native_box(100, 10, 140, 18)
    version = TableCell(
        cell_id="table-1-r2-c1",
        row_index=2,
        column_index=1,
        text="VersionRAG",
        bounding_box=native_box(100, 20, 150, 28),
    )
    table.cells.append(version)
    table.rows[1].cells.append(version)
    before = table.model_dump(mode="json")

    repaired = DoclingDocumentParser()._repair_table_from_native_geometry(table, words)

    assert repaired is False
    assert table.model_dump(mode="json") == before


def test_native_geometry_does_not_fill_genuine_blank_cell() -> None:
    table, words = malformed_two_row_group()
    table.cells.pop(1)
    table.rows[0].cells.pop(1)
    before = table.model_dump(mode="json")

    repaired = DoclingDocumentParser()._repair_table_from_native_geometry(table, words)

    assert repaired is False
    assert table.model_dump(mode="json") == before
    assert not any(cell.row_index == 2 and cell.column_index == 0 for cell in table.cells)


def test_native_geometry_keeps_independent_row_groups_separate() -> None:
    table, words = malformed_two_row_group()
    second_group = [
        TableCell(
            cell_id="table-1-r3-c0",
            row_index=3,
            column_index=0,
            text="Model B",
            row_span=2,
            bounding_box=native_box(10, 34, 70, 44),
        ),
        TableCell(
            cell_id="table-1-r3-c1",
            row_index=3,
            column_index=1,
            text="GraphRAG",
            bounding_box=native_box(100, 30, 140, 38),
        ),
        TableCell(
            cell_id="table-1-r4-c1",
            row_index=4,
            column_index=1,
            text="VersionRAG",
            bounding_box=native_box(100, 40, 150, 48),
        ),
    ]
    table.cells.extend(second_group)
    table.rows.extend(
        [
            TableRow(row_index=3, cells=second_group[:2]),
            TableRow(row_index=4, cells=second_group[2:]),
        ]
    )
    words.extend(
        [
            native_word("Model", 10, 34, 42, 44),
            native_word("B", 45, 34, 52, 44),
            native_word("GraphRAG", 100, 30, 140, 38),
            native_word("VersionRAG", 100, 40, 150, 48),
        ]
    )

    repaired = DoclingDocumentParser()._repair_table_from_native_geometry(table, words)

    assert repaired is True
    model_b = next(cell for cell in table.cells if cell.text == "Model B")
    assert (model_b.row_index, model_b.row_span) == (3, 2)
    assert not any(cell.row_index == 2 and cell.text == "Model B" for cell in table.cells)


def test_native_geometry_abstains_when_cross_row_alignment_is_ambiguous() -> None:
    table, words = malformed_two_row_group()
    table.cells[0].bounding_box = native_box(10, 10, 70, 18)
    table.rows[0].cells[0].bounding_box = native_box(10, 10, 70, 18)
    before = table.model_dump(mode="json")

    repaired = DoclingDocumentParser()._repair_table_from_native_geometry(table, words)

    assert repaired is False
    assert table.model_dump(mode="json") == before


def test_native_geometry_abstains_when_cell_coordinates_are_missing() -> None:
    table, words = malformed_two_row_group()
    table.cells[0].bounding_box = None
    table.rows[0].cells[0].bounding_box = None
    before = table.model_dump(mode="json")

    repaired = DoclingDocumentParser()._repair_table_from_native_geometry(table, words)

    assert repaired is False
    assert table.model_dump(mode="json") == before


def test_native_geometry_abstains_when_native_text_does_not_match() -> None:
    table, words = malformed_two_row_group()
    words[3] = native_word("OtherRAG", 100, 20, 150, 28)
    before = table.model_dump(mode="json")

    repaired = DoclingDocumentParser()._repair_table_from_native_geometry(table, words)

    assert repaired is False
    assert table.model_dump(mode="json") == before
