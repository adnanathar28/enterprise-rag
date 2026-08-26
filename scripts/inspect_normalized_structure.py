from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from brd_knowledge.schemas.document import DocumentBlock, ParsedDocument, ParsedImage
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.table import ParsedTable

NONE = "NONE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an existing normalized ParsedDocument JSON file."
    )
    parser.add_argument("parsed_document_path", type=Path)
    return parser.parse_args()


def load_parsed_document(path: Path) -> ParsedDocument:
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Parsed document JSON does not exist: {resolved_path}")
    return ParsedDocument.model_validate_json(resolved_path.read_text(encoding="utf-8"))


def _walk_sections(
    sections: list[DocumentSection],
    parent_id: str | None = None,
) -> list[tuple[DocumentSection, str | None]]:
    flattened = []
    for section in sections:
        flattened.append((section, parent_id))
        flattened.extend(_walk_sections(section.child_sections, section.section_id))
    return flattened


def _section_membership(
    parsed_document: ParsedDocument,
) -> tuple[dict[str, DocumentSection], dict[str, DocumentSection]]:
    block_sections: dict[str, DocumentSection] = {}
    table_sections: dict[str, DocumentSection] = {}
    for section, _ in _walk_sections(parsed_document.sections):
        if section.heading_block is not None:
            block_sections.setdefault(section.heading_block.block_id, section)
        for block in section.blocks:
            block_sections.setdefault(block.block_id, section)
        for paragraph in section.paragraphs:
            block_sections.setdefault(paragraph.block_id, section)
        for table in section.tables:
            table_sections.setdefault(table.table_id, section)
    return block_sections, table_sections


def _quoted(value: str, limit: int = 180) -> str:
    compact = " ".join(value.split())
    if len(compact) > limit:
        compact = f"{compact[: limit - 3]}..."
    return json.dumps(compact, ensure_ascii=False)


def _order(value: int | None) -> str:
    return str(value) if value is not None else NONE


def _section_fields(section: DocumentSection | None) -> list[str]:
    if section is None:
        return [f"     section_id={NONE}", f"     section_title={NONE}"]
    return [
        f"     section_id={section.section_id}",
        f"     section_title={_quoted(section.title)}",
    ]


def _render_block(block: DocumentBlock, section: DocumentSection | None) -> list[str]:
    return [
        f"[{_order(block.reading_order_index)}] {block.block_type}",
        f"     block_id={block.block_id}",
        *_section_fields(section),
        f"     page={block.page_number}",
        f"     text={_quoted(block.text)}",
    ]


def _first_row_preview(table: ParsedTable) -> str | None:
    if not table.rows:
        return None
    cells = sorted(table.rows[0].cells, key=lambda cell: cell.column_index)
    text = " | ".join(" ".join(cell.text.split()) for cell in cells if cell.text.strip())
    return text or None


def _render_table(table: ParsedTable, section: DocumentSection | None) -> list[str]:
    lines = [
        f"[{_order(table.reading_order_index)}] table",
        f"     table_id={table.table_id}",
        *_section_fields(section),
        f"     page={table.page_number if table.page_number is not None else NONE}",
        f"     rows={len(table.rows)}",
        f"     cells={len(table.cells)}",
        f"     caption={_quoted(table.caption) if table.caption is not None else NONE}",
    ]
    preview = _first_row_preview(table)
    if preview is not None:
        lines.append(f"     first_row={_quoted(preview)}")
    return lines


def _render_image(image: ParsedImage) -> list[str]:
    return [
        f"[{_order(image.reading_order_index)}] image",
        f"     image_id={image.image_id}",
        f"     page={image.page_number}",
        f"     caption={_quoted(image.caption) if image.caption is not None else NONE}",
    ]


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _health_flags(
    parsed_document: ParsedDocument,
    block_sections: dict[str, DocumentSection],
    table_sections: dict[str, DocumentSection],
) -> list[str]:
    flags = []
    unsectioned_blocks = sorted(
        block.block_id
        for block in parsed_document.blocks
        if block.block_id not in block_sections
    )
    unsectioned_tables = sorted(
        table.table_id
        for table in parsed_document.tables
        if table.table_id not in table_sections
    )
    if unsectioned_blocks:
        flags.append(f"blocks with no section: {', '.join(unsectioned_blocks)}")
    if unsectioned_tables:
        flags.append(f"tables with no section: {', '.join(unsectioned_tables)}")

    flattened_sections = [section for section, _ in _walk_sections(parsed_document.sections)]
    levels = {section.level for section in flattened_sections}
    if len(flattened_sections) > 1 and len(levels) == 1:
        flags.append(f"all sections have the same level: {next(iter(levels))}")
    if flattened_sections and all(not section.child_sections for section in flattened_sections):
        flags.append("child_sections is empty for every section")

    duplicate_blocks = _duplicates([block.block_id for block in parsed_document.blocks])
    duplicate_tables = _duplicates([table.table_id for table in parsed_document.tables])
    if duplicate_blocks:
        flags.append(f"duplicate block IDs: {', '.join(duplicate_blocks)}")
    if duplicate_tables:
        flags.append(f"duplicate table IDs: {', '.join(duplicate_tables)}")

    missing_order = sorted(
        [
            f"block:{block.block_id}"
            for block in parsed_document.blocks
            if block.reading_order_index is None
        ]
        + [
            f"table:{table.table_id}"
            for table in parsed_document.tables
            if table.reading_order_index is None
        ]
    )
    if missing_order:
        flags.append(f"missing reading_order_index: {', '.join(missing_order)}")

    missing_source = sorted(
        [f"block:{block.block_id}" for block in parsed_document.blocks if block.source is None]
        + [f"table:{table.table_id}" for table in parsed_document.tables if table.source is None]
    )
    if missing_source:
        flags.append(f"missing provenance/source: {', '.join(missing_source)}")

    failed_pages = [
        str(page.page_number) for page in parsed_document.pages if page.parse_status == "failed"
    ]
    if failed_pages:
        flags.append(f"failed pages: {', '.join(failed_pages)}")
    empty_pages = [
        str(page.page_number)
        for page in parsed_document.pages
        if page.parse_status != "failed" and not page.blocks and not page.tables and not page.images
    ]
    if empty_pages:
        flags.append(f"non-failed empty pages: {', '.join(empty_pages)}")
    return flags


def render_normalized_structure(parsed_document: ParsedDocument) -> str:
    block_sections, table_sections = _section_membership(parsed_document)
    lines = [
        f"DOCUMENT {parsed_document.metadata.document_id}",
        f"filename={parsed_document.metadata.filename}",
        "",
    ]

    for page in sorted(parsed_document.pages, key=lambda item: item.page_number):
        lines.extend([f"PAGE {page.page_number}", ""])
        items: list[tuple[int, int, str, object]] = []
        items.extend(
            (
                block.reading_order_index if block.reading_order_index is not None else 10**9,
                0,
                block.block_id,
                block,
            )
            for block in page.blocks
        )
        items.extend(
            (
                table.reading_order_index if table.reading_order_index is not None else 10**9,
                1,
                table.table_id,
                table,
            )
            for table in page.tables
        )
        items.extend(
            (
                image.reading_order_index if image.reading_order_index is not None else 10**9,
                2,
                image.image_id,
                image,
            )
            for image in page.images
        )
        for _, _, _, item in sorted(items, key=lambda entry: entry[:3]):
            if isinstance(item, DocumentBlock):
                lines.extend(_render_block(item, block_sections.get(item.block_id)))
            elif isinstance(item, ParsedTable):
                lines.extend(_render_table(item, table_sections.get(item.table_id)))
            elif isinstance(item, ParsedImage):
                lines.extend(_render_image(item))
            lines.append("")

    lines.extend(["SECTION SUMMARY", ""])
    flattened_sections = _walk_sections(parsed_document.sections)
    if not flattened_sections:
        lines.extend(["NONE", ""])
    for section, parent_id in flattened_sections:
        lines.extend(
            [
                f"section_id={section.section_id}",
                f"     title={_quoted(section.title)}",
                f"     level={section.level}",
                f"     page_start={section.page_start}",
                f"     page_end={section.page_end if section.page_end is not None else NONE}",
                f"     parent_section={parent_id or NONE}",
                f"     blocks={len(section.blocks)}",
                f"     tables={len(section.tables)}",
                f"     child_sections={len(section.child_sections)}",
                "",
            ]
        )

    lines.extend(["HEALTH FLAGS", ""])
    flags = _health_flags(parsed_document, block_sections, table_sections)
    lines.extend([f"- {flag}" for flag in flags] if flags else ["- none"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    print(render_normalized_structure(load_parsed_document(args.parsed_document_path)), end="")


if __name__ == "__main__":
    main()
