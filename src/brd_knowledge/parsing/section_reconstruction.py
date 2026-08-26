from __future__ import annotations

import re
from collections.abc import Sequence

from brd_knowledge.schemas.document import DocumentBlock, TextBlock
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable

NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*(?:(?P<nested>\d+(?:\.\d+)+)(?:[.)])?(?=\s|$)|(?P<top>\d+)[.)])\s*"
)
FOOTER_BRAND_TEXT = frozenset({"F", "ERO", "FERO", "FEPO", "EEPO", "CCD", "CCC"})


def is_semantic_block(block: DocumentBlock) -> bool:
    return block.block_type != "page_footer" and not _is_footer_brand_artifact(block)


def _is_footer_brand_artifact(block: DocumentBlock) -> bool:
    bounding_box = block.bounding_box
    return (
        block.block_type == "text"
        and block.text.strip().upper() in FOOTER_BRAND_TEXT
        and bounding_box is not None
        and bounding_box.x0 >= 450
        and min(bounding_box.y0, bounding_box.y1) <= 60
    )


def infer_heading_level(heading_text: str) -> int:
    """Infer numbered heading depth; keep unnumbered headings as level-one roots."""
    match = NUMBERED_HEADING_PATTERN.match(heading_text)
    if match is None:
        return 1
    nested_number = match.group("nested")
    if nested_number is not None:
        return len(nested_number.split("."))
    return 1


def rebuild_document_sections(
    document_id: str,
    blocks: Sequence[DocumentBlock],
    tables: Sequence[ParsedTable] = (),
) -> list[DocumentSection]:
    ordered_items = sorted(
        [block for block in blocks if is_semantic_block(block)] + list(tables),
        key=_content_order_key,
    )
    root_sections: list[DocumentSection] = []
    active_section: DocumentSection | None = None
    numbered_section_stack: list[DocumentSection] = []

    for item in ordered_items:
        if isinstance(item, ParsedTable):
            if active_section is not None:
                active_section.tables.append(item)
                if item.page_number is not None:
                    active_section.page_end = max(
                        active_section.page_end or item.page_number,
                        item.page_number,
                    )
            continue

        block = item
        if block.block_type == "section_header":
            level = infer_heading_level(block.text)
            is_numbered = NUMBERED_HEADING_PATTERN.match(block.text) is not None
            section_id = f"section-{block.block_id}"
            heading_block = _as_text_block(block)
            section = DocumentSection(
                section_id=section_id,
                title=block.text,
                level=level,
                page_start=block.page_number,
                page_end=block.page_number,
                source=SourceReference(
                    document_id=document_id,
                    page_number=block.page_number,
                    section_id=section_id,
                    block_id=block.block_id,
                    reading_order_index=block.reading_order_index,
                    text_excerpt=_excerpt(block.text),
                    bounding_box=block.bounding_box,
                ),
                heading_block=heading_block,
                blocks=[block],
                paragraphs=[heading_block],
            )

            if not is_numbered:
                # Docling does not provide dependable hierarchy for unnumbered
                # headings. Keep them as level-one roots without allowing them to
                # erase the active numbered ancestry.
                root_sections.append(section)
            else:
                while numbered_section_stack and numbered_section_stack[-1].level >= level:
                    numbered_section_stack.pop()
                if numbered_section_stack:
                    numbered_section_stack[-1].child_sections.append(section)
                else:
                    root_sections.append(section)
                numbered_section_stack.append(section)
            active_section = section
            continue

        if active_section is not None:
            active_section.blocks.append(block)
            active_section.paragraphs.append(_as_text_block(block))
            active_section.page_end = max(
                active_section.page_end or block.page_number,
                block.page_number,
            )

    for section in root_sections:
        _extend_page_end_through_children(section)
    return root_sections


def _content_order_key(item: DocumentBlock | ParsedTable) -> tuple[int, int, int, str]:
    is_table = isinstance(item, ParsedTable)
    page_number = item.page_number
    if isinstance(item, ParsedTable):
        item_id = item.table_id
    else:
        item_id = item.block_id
    return (
        page_number if page_number is not None else 10**9,
        item.reading_order_index if item.reading_order_index is not None else 10**9,
        1 if is_table else 0,
        item_id,
    )


def _as_text_block(block: DocumentBlock) -> TextBlock:
    return TextBlock.model_validate(block.model_dump())


def _extend_page_end_through_children(section: DocumentSection) -> int:
    page_end = section.page_end or section.page_start
    for child_section in section.child_sections:
        page_end = max(page_end, _extend_page_end_through_children(child_section))
    section.page_end = page_end
    return page_end


def _excerpt(text: str, limit: int = 240) -> str:
    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    return f"{compact_text[: limit - 3]}..."
