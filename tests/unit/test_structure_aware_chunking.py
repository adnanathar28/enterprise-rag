from brd_knowledge.chunking import StructureAwareChunker
from brd_knowledge.schemas.document import DocumentMetadata, ParsedDocument, TextBlock
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable, TableCell, TableRow


def block(
    block_id: str,
    text: str,
    page: int,
    order: int,
    block_type: str = "paragraph",
) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        page_number=page,
        reading_order_index=order,
        block_type=block_type,
        text=text,
        source=SourceReference(
            document_id="doc-1",
            page_number=page,
            block_id=block_id,
            reading_order_index=order,
        ),
    )


def table(
    table_id: str,
    page: int,
    order: int,
    *,
    native_text: str | None = None,
    coverage: float | None = None,
    quality_notes: list[str] | None = None,
) -> ParsedTable:
    cells = [
        TableCell(row_index=0, column_index=0, text="Field", is_header=True),
        TableCell(row_index=0, column_index=1, text="Value", is_header=True),
        TableCell(row_index=1, column_index=0, text="Priority"),
        TableCell(row_index=1, column_index=1, text="Must"),
    ]
    return ParsedTable(
        table_id=table_id,
        page_number=page,
        page_numbers=[page],
        reading_order_index=order,
        rows=[
            TableRow(row_index=0, cells=cells[:2]),
            TableRow(row_index=1, cells=cells[2:]),
        ],
        cells=cells,
        source=SourceReference(
            document_id="doc-1",
            page_number=page,
            table_id=table_id,
            reading_order_index=order,
        ),
        native_text=native_text,
        native_text_coverage=coverage,
        quality_notes=quality_notes or [],
    )


def document_with_sections(sections: list[DocumentSection]) -> ParsedDocument:
    all_blocks: list[TextBlock] = []
    all_tables: list[ParsedTable] = []

    def collect(section: DocumentSection) -> None:
        all_blocks.extend(TextBlock.model_validate(item.model_dump()) for item in section.blocks)
        all_tables.extend(section.tables)
        for child in section.child_sections:
            collect(child)

    for section in sections:
        collect(section)
    return ParsedDocument(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="sample.pdf",
            file_type="pdf",
            page_count=3,
        ),
        sections=sections,
        blocks=all_blocks,
        tables=all_tables,
    )


def render_table(rows: list[TableRow], cells: list[TableCell]) -> str:
    parsed_table = ParsedTable(
        table_id="table-spans",
        page_number=1,
        page_numbers=[1],
        reading_order_index=0,
        rows=rows,
        cells=cells,
        source=SourceReference(document_id="doc-1", page_number=1, table_id="table-spans"),
        native_text_coverage=1.0,
    )
    section = DocumentSection(
        section_id="section-spans",
        title="Table Spans",
        level=1,
        page_start=1,
        tables=[parsed_table],
    )
    return StructureAwareChunker().chunk(document_with_sections([section]))[0].text


def test_combines_prose_within_section_and_preserves_provenance() -> None:
    heading = block("heading-1", "1) Scope", 1, 0, "section_header")
    first = block("block-1", "The system tracks inventory.", 1, 1)
    second = block("block-2", "Changes require approval.", 1, 2)
    section = DocumentSection(
        section_id="section-1",
        title="1) Scope",
        level=1,
        page_start=1,
        page_end=1,
        heading_block=heading,
        blocks=[heading, first, second],
    )

    chunks = StructureAwareChunker().chunk(document_with_sections([section]))

    assert len(chunks) == 1
    assert chunks[0].text == (
        "Section: 1) Scope\n\n"
        "The system tracks inventory.\n\nChanges require approval."
    )
    assert chunks[0].source_block_ids == ["heading-1", "block-1", "block-2"]
    assert [source.block_id for source in chunks[0].provenance] == [
        "heading-1",
        "block-1",
        "block-2",
    ]


def test_never_combines_unrelated_sections() -> None:
    first_heading = block("heading-1", "1) First", 1, 0, "section_header")
    second_heading = block("heading-2", "2) Second", 1, 2, "section_header")
    first = DocumentSection(
        section_id="section-1",
        title="1) First",
        level=1,
        page_start=1,
        blocks=[first_heading, block("block-1", "First content", 1, 1)],
        heading_block=first_heading,
    )
    second = DocumentSection(
        section_id="section-2",
        title="2) Second",
        level=1,
        page_start=1,
        blocks=[second_heading, block("block-2", "Second content", 1, 3)],
        heading_block=second_heading,
    )

    chunks = StructureAwareChunker().chunk(document_with_sections([first, second]))

    assert [chunk.section_id for chunk in chunks] == ["section-1", "section-2"]
    assert "Second content" not in chunks[0].text
    assert "First content" not in chunks[1].text


def test_multi_page_section_has_complete_page_range() -> None:
    heading = block("heading-1", "1) Scope", 1, 0, "section_header")
    section = DocumentSection(
        section_id="section-1",
        title="1) Scope",
        level=1,
        page_start=1,
        page_end=2,
        heading_block=heading,
        blocks=[
            heading,
            block("block-1", "Page one", 1, 1),
            block("block-2", "Page two", 2, 0),
        ],
    )

    chunk = StructureAwareChunker().chunk(document_with_sections([section]))[0]

    assert (chunk.page_start, chunk.page_end) == (1, 2)


def test_healthy_table_uses_structured_rows() -> None:
    heading = block("heading-1", "1) Requirements", 1, 0, "section_header")
    parsed_table = table("table-1", 1, 1, native_text="Ignored native text", coverage=1.0)
    section = DocumentSection(
        section_id="section-1",
        title="1) Requirements",
        level=1,
        page_start=1,
        heading_block=heading,
        blocks=[heading],
        tables=[parsed_table],
    )

    chunk = StructureAwareChunker().chunk(document_with_sections([section]))[0]

    assert chunk.content_type == "table"
    assert chunk.text.endswith("Field | Value\nPriority | Must")
    assert "Ignored native text" not in chunk.text
    assert chunk.source_table_ids == ["table-1"]


def test_explicit_row_span_repeats_label_in_covered_row() -> None:
    model = TableCell(row_index=1, column_index=0, text="Model A", row_span=2)
    cells = [
        TableCell(row_index=0, column_index=0, text="Model", is_header=True),
        TableCell(row_index=0, column_index=1, text="Approach", is_header=True),
        TableCell(row_index=0, column_index=2, text="Input", is_header=True),
        model,
        TableCell(row_index=1, column_index=1, text="GraphRAG"),
        TableCell(row_index=1, column_index=2, text="100"),
        TableCell(row_index=2, column_index=1, text="VersionRAG"),
        TableCell(row_index=2, column_index=2, text="50"),
    ]
    rows = [
        TableRow(row_index=0, cells=cells[:3]),
        TableRow(row_index=1, cells=cells[3:6]),
        TableRow(row_index=2, cells=cells[6:]),
    ]

    rendered = render_table(rows, cells)

    assert rendered.endswith(
        "Model | Approach | Input\n"
        "Model A | GraphRAG | 100\n"
        "Model A | VersionRAG | 50"
    )


def test_independent_row_spans_do_not_leak_between_groups() -> None:
    model_a = TableCell(row_index=0, column_index=0, text="Model A", row_span=2)
    graph_a = TableCell(row_index=0, column_index=1, text="GraphRAG")
    version_a = TableCell(row_index=1, column_index=1, text="VersionRAG")
    model_b = TableCell(row_index=2, column_index=0, text="Model B", row_span=2)
    graph_b = TableCell(row_index=2, column_index=1, text="GraphRAG")
    version_b = TableCell(row_index=3, column_index=1, text="VersionRAG")
    cells = [model_a, graph_a, version_a, model_b, graph_b, version_b]
    rows = [
        TableRow(row_index=0, cells=[model_a, graph_a]),
        TableRow(row_index=1, cells=[version_a]),
        TableRow(row_index=2, cells=[model_b, graph_b]),
        TableRow(row_index=3, cells=[version_b]),
    ]

    rendered = render_table(rows, cells)

    assert rendered.endswith(
        "Model A | GraphRAG\n"
        "Model A | VersionRAG\n"
        "Model B | GraphRAG\n"
        "Model B | VersionRAG"
    )


def test_explicit_column_span_repeats_merged_header_across_covered_columns() -> None:
    name = TableCell(row_index=0, column_index=0, text="Name", is_header=True)
    dimensions = TableCell(
        row_index=0,
        column_index=1,
        text="Dimensions",
        column_span=2,
        is_header=True,
    )
    small = TableCell(row_index=1, column_index=0, text="Small")
    width = TableCell(row_index=1, column_index=1, text="20")
    height = TableCell(row_index=1, column_index=2, text="15")
    cells = [name, dimensions, small, width, height]
    rows = [
        TableRow(row_index=0, cells=[name, dimensions]),
        TableRow(row_index=1, cells=[small, width, height]),
    ]

    rendered = render_table(rows, cells)

    assert rendered.endswith("Name | Dimensions | Dimensions\nSmall | 20 | 15")


def test_genuinely_empty_cell_is_not_forward_filled_without_a_span() -> None:
    label = TableCell(row_index=0, column_index=0, text="Label")
    first_value = TableCell(row_index=0, column_index=1, text="First")
    second_value = TableCell(row_index=1, column_index=1, text="Second")
    cells = [label, first_value, second_value]
    rows = [
        TableRow(row_index=0, cells=[label, first_value]),
        TableRow(row_index=1, cells=[second_value]),
    ]

    rendered = render_table(rows, cells)

    assert rendered.endswith("Label | First\n | Second")


def test_flagged_table_uses_native_text_and_carries_quality_warning() -> None:
    warning = "Structured table text is incomplete; use native_text as a fallback."
    parsed_table = table(
        "table-1",
        2,
        0,
        native_text="Recovered requirement three and requirement four.",
        coverage=0.4,
        quality_notes=[warning],
    )
    section = DocumentSection(
        section_id="section-1",
        title="1) Requirements",
        level=1,
        page_start=2,
        tables=[parsed_table],
    )

    chunk = StructureAwareChunker().chunk(document_with_sections([section]))[0]

    assert chunk.text.endswith("Recovered requirement three and requirement four.")
    assert "Field | Value" not in chunk.text
    assert chunk.quality_notes == [warning]


def test_chunk_ids_are_deterministic() -> None:
    heading = block("heading-1", "1) Scope", 1, 0, "section_header")
    section = DocumentSection(
        section_id="section-1",
        title="1) Scope",
        level=1,
        page_start=1,
        heading_block=heading,
        blocks=[heading, block("block-1", "Stable content", 1, 1)],
    )
    document = document_with_sections([section])
    chunker = StructureAwareChunker()

    assert [chunk.chunk_id for chunk in chunker.chunk(document)] == [
        chunk.chunk_id for chunk in chunker.chunk(document)
    ]


def test_oversized_block_is_split_deterministically_with_source_mapping() -> None:
    heading = block("heading-1", "1) Scope", 1, 0, "section_header")
    oversized = block("block-1", "A" * 120, 1, 1)
    section = DocumentSection(
        section_id="section-1",
        title="1) Scope",
        level=1,
        page_start=1,
        heading_block=heading,
        blocks=[heading, oversized],
    )

    chunks = StructureAwareChunker(max_characters=60).chunk(document_with_sections([section]))

    assert len(chunks) == 3
    assert all(len(chunk.text) <= 60 for chunk in chunks)
    assert chunks[0].source_block_ids == ["heading-1", "block-1"]
    assert all(chunk.source_block_ids == ["block-1"] for chunk in chunks[1:])


def test_sources_are_not_accidentally_duplicated() -> None:
    heading = block("heading-1", "1) Scope", 1, 0, "section_header")
    first_table = table("table-1", 1, 2)
    section = DocumentSection(
        section_id="section-1",
        title="1) Scope",
        level=1,
        page_start=1,
        heading_block=heading,
        blocks=[heading, block("block-1", "Content", 1, 1)],
        tables=[first_table],
    )

    chunks = StructureAwareChunker().chunk(document_with_sections([section]))
    block_ids = [block_id for chunk in chunks for block_id in chunk.source_block_ids]
    table_ids = [table_id for chunk in chunks for table_id in chunk.source_table_ids]

    assert block_ids == ["heading-1", "block-1"]
    assert table_ids == ["table-1"]


def test_unsectioned_content_is_preserved() -> None:
    orphan = block("orphan", "Cover page content", 1, 0)
    document = document_with_sections([])
    document.blocks = [orphan]

    chunk = StructureAwareChunker().chunk(document)[0]

    assert chunk.section_id is None
    assert chunk.section_path == []
    assert chunk.text == "Cover page content"
    assert chunk.source_block_ids == ["orphan"]


def test_heading_only_section_keeps_context_without_repeating_title() -> None:
    heading = block("heading-1", "1) Empty Section", 1, 0, "section_header")
    section = DocumentSection(
        section_id="section-1",
        title="1) Empty Section",
        level=1,
        page_start=1,
        heading_block=heading,
        blocks=[heading],
    )

    chunk = StructureAwareChunker().chunk(document_with_sections([section]))[0]

    assert chunk.text == "Section: 1) Empty Section"
    assert chunk.source_block_ids == ["heading-1"]
