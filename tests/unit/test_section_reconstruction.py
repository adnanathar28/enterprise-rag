from brd_knowledge.parsing.section_reconstruction import rebuild_document_sections
from brd_knowledge.parsing.split_pages import merge_parsed_documents
from brd_knowledge.schemas.document import (
    DocumentMetadata,
    Page,
    ParsedDocument,
    ParserMetadata,
    TextBlock,
)
from brd_knowledge.schemas.source import BoundingBox, SourceReference
from brd_knowledge.schemas.table import ParsedTable


def block(
    block_id: str,
    text: str,
    page_number: int,
    reading_order_index: int,
    block_type: str = "text",
) -> TextBlock:
    bounding_box = BoundingBox(
        page_number=page_number,
        x0=10,
        y0=20,
        x1=100,
        y1=40,
        coordinate_origin="top_left",
    )
    return TextBlock(
        block_id=block_id,
        page_number=page_number,
        text=text,
        block_type=block_type,
        reading_order_index=reading_order_index,
        bounding_box=bounding_box,
        source=SourceReference(
            document_id="doc-1",
            page_number=page_number,
            parser_item_id=f"#/texts/{block_id}",
            block_id=block_id,
            reading_order_index=reading_order_index,
            bounding_box=bounding_box,
        ),
    )


def table(table_id: str, page_number: int, reading_order_index: int) -> ParsedTable:
    bounding_box = BoundingBox(
        page_number=page_number,
        x0=10,
        y0=50,
        x1=100,
        y1=90,
        coordinate_origin="top_left",
    )
    return ParsedTable(
        table_id=table_id,
        page_number=page_number,
        page_numbers=[page_number],
        reading_order_index=reading_order_index,
        bounding_box=bounding_box,
        source=SourceReference(
            document_id="doc-1",
            page_number=page_number,
            parser_item_id=f"#/tables/{table_id}",
            table_id=table_id,
            reading_order_index=reading_order_index,
            bounding_box=bounding_box,
        ),
    )
def test_rebuilds_numbered_hierarchy_and_section_boundaries_across_pages() -> None:
    master = block("master", "1) Master Data Setup", 1, 1, "section_header")
    master_intro = block("master-intro", "Master data introduction.", 1, 2)
    vehicle = block("vehicle", "1.1 Vehicle Master Configuration", 2, 1, "section_header")
    vehicle_text = block("vehicle-text", "Vehicle configuration details.", 2, 2)
    permit = block("permit", "1.2 Permit Configuration", 2, 3, "section_header")
    permit_text = block("permit-text", "Permit configuration details.", 2, 4)
    orders = block("orders", "2) Order Management", 3, 1, "section_header")
    status = block("status", "2.1 Order Status Model", 3, 2, "section_header")
    status_text = block("status-text", "Order status details.", 3, 3)

    sections = rebuild_document_sections(
        "doc-1",
        [
            status_text,
            permit,
            master_intro,
            orders,
            vehicle_text,
            master,
            status,
            permit_text,
            vehicle,
        ],
    )

    assert [section.title for section in sections] == [
        "1) Master Data Setup",
        "2) Order Management",
    ]
    assert [child.title for child in sections[0].child_sections] == [
        "1.1 Vehicle Master Configuration",
        "1.2 Permit Configuration",
    ]
    assert [child.title for child in sections[1].child_sections] == [
        "2.1 Order Status Model"
    ]
    assert sections[0].level == 1
    assert sections[0].child_sections[0].level == 2
    assert sections[0].page_start == 1
    assert sections[0].page_end == 2
    assert [item.block_id for item in sections[0].blocks] == ["master", "master-intro"]
    assert [item.block_id for item in sections[0].child_sections[0].blocks] == [
        "vehicle",
        "vehicle-text",
    ]


def test_reconstruction_preserves_heading_id_and_provenance() -> None:
    heading = block("page_4_texts-3", "1.1 Vehicle Master", 4, 7, "section_header")

    sections = rebuild_document_sections("doc-1", [heading])

    section = sections[0]
    assert section.section_id == "section-page_4_texts-3"
    assert section.heading_block is not None
    assert section.heading_block.block_id == heading.block_id
    assert section.heading_block.source == heading.source
    assert section.heading_block.bounding_box == heading.bounding_box
    assert section.source is not None
    assert section.source.block_id == heading.block_id
    assert section.source.bounding_box == heading.bounding_box


def test_merge_excludes_page_footer_from_pages_blocks_and_sections() -> None:
    heading = block("heading", "1) Scope", 1, 1, "section_header")
    paragraph = block("paragraph", "Scope details.", 1, 2)
    footer = block("footer", "12", 1, 0, "page_footer")
    parsed_page = ParsedDocument(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="sample.pdf",
            file_type="pdf",
            page_count=1,
        ),
        parser_metadata=ParserMetadata(parser_name="docling", parse_status="success"),
        pages=[
            Page(
                page_number=1,
                parse_status="success",
                blocks=[footer, paragraph, heading],
            )
        ],
        blocks=[footer, paragraph, heading],
        paragraphs=[footer, paragraph, heading],
    )

    merged = merge_parsed_documents([parsed_page])

    assert [item.block_id for item in merged.blocks] == ["heading", "paragraph"]
    assert [item.block_id for item in merged.pages[0].blocks] == ["heading", "paragraph"]
    assert [item.block_id for item in merged.sections[0].blocks] == ["heading", "paragraph"]
    assert all(item.block_type != "page_footer" for item in merged.paragraphs)


def test_footer_brand_artifact_is_excluded_without_filtering_body_text() -> None:
    heading = block("heading", "Scope", 1, 1, "section_header")
    body_fero = block("body-fero", "FERO", 1, 2)
    footer_fero = block("footer-fero", "FEPO", 1, 3)
    footer_fero.bounding_box = BoundingBox(
        page_number=1,
        x0=466,
        y0=53,
        x1=524,
        y1=44,
        coordinate_origin="unknown",
    )

    sections = rebuild_document_sections("doc-1", [footer_fero, body_fero, heading])

    assert [item.block_id for item in sections[0].blocks] == ["heading", "body-fero"]


def test_unnumbered_headings_remain_level_one_roots() -> None:
    overview = block("overview", "Overview", 1, 1, "section_header")
    details = block("details", "Details", 1, 2, "section_header")

    sections = rebuild_document_sections("doc-1", [overview, details])

    assert [section.level for section in sections] == [1, 1]
    assert all(not section.child_sections for section in sections)


def test_unnumbered_heading_does_not_replace_numbered_parent() -> None:
    master = block("master", "1)Master Data Setup", 1, 1, "section_header")
    package = block("package", "1.8 Package Configuration", 1, 2, "section_header")
    operational_rules = block("rules", "Operational Rules:", 2, 1, "section_header")
    rule_text = block("rule-text", "Invoices must not be split.", 2, 2)
    validation = block("validation", "1.9 Pre-Go-Live Validation", 2, 3, "section_header")
    validation_text = block("validation-text", "Validate all master data.", 2, 4)
    orders = block("orders", "2) Order Management", 3, 1, "section_header")
    status = block("status", "2.1 Order Status Model", 3, 2, "section_header")

    sections = rebuild_document_sections(
        "doc-1",
        [
            validation_text,
            operational_rules,
            package,
            status,
            master,
            rule_text,
            orders,
            validation,
        ],
    )

    assert [section.title for section in sections] == [
        "1)Master Data Setup",
        "Operational Rules:",
        "2) Order Management",
    ]
    assert [child.title for child in sections[0].child_sections] == [
        "1.8 Package Configuration",
        "1.9 Pre-Go-Live Validation",
    ]
    assert [block.block_id for block in sections[1].blocks] == ["rules", "rule-text"]
    assert [block.block_id for block in sections[0].child_sections[1].blocks] == [
        "validation",
        "validation-text",
    ]
    assert [child.title for child in sections[2].child_sections] == [
        "2.1 Order Status Model"
    ]


def test_tables_attach_to_active_section_in_document_order_across_pages() -> None:
    scope = block("scope", "1) Scope", 1, 1, "section_header")
    scope_table = table("scope-table", 1, 3)
    continuation_table = table("scope-table-continuation", 2, 1)
    requirements = block("requirements", "2) Requirements", 2, 2, "section_header")
    requirements_table = table("requirements-table", 2, 3)

    sections = rebuild_document_sections(
        "doc-1",
        [requirements, scope],
        [requirements_table, continuation_table, scope_table],
    )

    assert [item.table_id for item in sections[0].tables] == [
        "scope-table",
        "scope-table-continuation",
    ]
    assert sections[0].page_end == 2
    assert [item.table_id for item in sections[1].tables] == ["requirements-table"]
    assert sections[0].tables[0].source == scope_table.source
    assert sections[0].tables[0].table_id == scope_table.table_id


def test_table_before_first_heading_remains_unassociated() -> None:
    preface_table = table("preface-table", 1, 1)
    scope = block("scope", "Scope", 1, 2, "section_header")

    sections = rebuild_document_sections("doc-1", [scope], [preface_table])

    assert len(sections) == 1
    assert sections[0].tables == []
