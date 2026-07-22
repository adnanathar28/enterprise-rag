from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brd_knowledge.parsing.docling_parser import DoclingDocumentParser
from brd_knowledge.schemas.document import (
    DocumentBlock,
    ParsedDocument,
    ParsedImage,
    ParserMetadata,
    ParseStatus,
)
from brd_knowledge.schemas.requirement import ExtractedRequirement
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable, TableCell

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a BRD into the internal ParsedDocument JSON schema."
    )
    parser.add_argument("document_path", type=Path, help="Path to an approved BRD PDF or DOCX.")
    parser.add_argument(
        "--page-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional inclusive 1-based page range to process.",
    )
    parser.add_argument(
        "--split-pages",
        action="store_true",
        help="Process each page in --page-range separately and merge one ParsedDocument.",
    )
    return parser.parse_args()


def validate_document_path(document_path: Path) -> Path:
    resolved_path = document_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Document does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Document path is not a file: {resolved_path}")
    if resolved_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(
            f"Unsupported file type '{resolved_path.suffix}'. Expected one of: {supported}"
        )
    return resolved_path


def validate_page_range(page_range: list[int] | None) -> tuple[int, int] | None:
    if page_range is None:
        return None

    start, end = page_range
    if start < 1 or end < 1:
        raise ValueError("Page range values must be positive 1-based page numbers.")
    if start > end:
        raise ValueError(f"Invalid page range: start page {start} is after end page {end}.")
    return start, end


def validate_split_pages(split_pages: bool, page_range: tuple[int, int] | None) -> None:
    if split_pages and page_range is None:
        raise ValueError("--split-pages requires --page-range START END.")


def build_output_dir(document_path: Path, page_range: tuple[int, int] | None) -> Path:
    output_name = document_path.stem
    if page_range is not None:
        output_name = f"{output_name}_pages_{page_range[0]}_{page_range[1]}"
    return Path("data/outputs") / output_name


def build_split_output_dir(document_path: Path, page_range: tuple[int, int]) -> Path:
    output_name = f"{document_path.stem}_pages_{page_range[0]}_{page_range[1]}_split"
    return Path("data/outputs") / output_name


def json_default(value: Any) -> str:
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def merge_parse_statuses(documents: list[ParsedDocument]) -> ParseStatus:
    statuses = [
        page.parse_status
        for document in documents
        for page in document.pages
        if page.parse_status not in {"not_started", "skipped"}
    ]
    if statuses and all(status == "success" for status in statuses):
        return "success"
    if statuses and all(status == "failed" for status in statuses):
        return "failed"
    return "partial_success"


def prefixed_id(page_number: int, value: str | None) -> str | None:
    if value is None:
        return None
    prefix = f"page_{page_number}_"
    if value.startswith(prefix):
        return value
    return f"{prefix}{value}"


def prefix_source_reference_ids(source: SourceReference | None, page_number: int) -> None:
    if source is None:
        return
    source.section_id = prefixed_id(page_number, source.section_id)
    source.block_id = prefixed_id(page_number, source.block_id)
    source.table_id = prefixed_id(page_number, source.table_id)
    source.cell_id = prefixed_id(page_number, source.cell_id)


def prefix_block_ids(block: DocumentBlock, page_number: int) -> None:
    block.block_id = prefixed_id(page_number, block.block_id) or block.block_id
    prefix_source_reference_ids(block.source, page_number)


def prefix_table_cell_ids(cell: TableCell, table_id: str, page_number: int) -> None:
    cell.cell_id = prefixed_id(page_number, cell.cell_id)
    prefix_source_reference_ids(cell.source, page_number)
    if cell.source is not None:
        cell.source.table_id = table_id


def prefix_table_ids(table: ParsedTable, page_number: int) -> None:
    table.table_id = prefixed_id(page_number, table.table_id) or table.table_id
    prefix_source_reference_ids(table.source, page_number)
    for cell in table.cells:
        prefix_table_cell_ids(cell, table.table_id, page_number)
    for row in table.rows:
        for cell in row.cells:
            prefix_table_cell_ids(cell, table.table_id, page_number)


def prefix_image_ids(image: ParsedImage, page_number: int) -> None:
    image.image_id = prefixed_id(page_number, image.image_id) or image.image_id
    prefix_source_reference_ids(image.source, page_number)


def prefix_section_ids(section: DocumentSection, page_number: int) -> None:
    section.section_id = prefixed_id(page_number, section.section_id) or section.section_id
    prefix_source_reference_ids(section.source, page_number)
    if section.heading_block is not None:
        prefix_block_ids(section.heading_block, page_number)
    for block in section.blocks:
        prefix_block_ids(block, page_number)
    for paragraph in section.paragraphs:
        prefix_block_ids(paragraph, page_number)
    for table in section.tables:
        prefix_table_ids(table, page_number)
    for child_section in section.child_sections:
        prefix_section_ids(child_section, page_number)


def prefix_requirement_source_ids(requirement: ExtractedRequirement, page_number: int) -> None:
    prefix_source_reference_ids(requirement.source, page_number)


def prefix_document_ids_for_split_page(
    document: ParsedDocument,
    page_number: int,
) -> ParsedDocument:
    prefixed_document = document.model_copy(deep=True)
    for block in prefixed_document.blocks:
        prefix_block_ids(block, page_number)
    for paragraph in prefixed_document.paragraphs:
        prefix_block_ids(paragraph, page_number)
    for table in prefixed_document.tables:
        prefix_table_ids(table, page_number)
    for image in prefixed_document.images:
        prefix_image_ids(image, page_number)
    for section in prefixed_document.sections:
        prefix_section_ids(section, page_number)
    for page in prefixed_document.pages:
        for block in page.blocks:
            prefix_block_ids(block, page_number)
        for table in page.tables:
            prefix_table_ids(table, page_number)
        for image in page.images:
            prefix_image_ids(image, page_number)
    return prefixed_document


def merge_parsed_documents(documents: list[ParsedDocument]) -> ParsedDocument:
    if not documents:
        raise ValueError("Cannot merge an empty list of parsed documents.")

    merged = documents[0].model_copy(deep=True)
    merged.pages = []
    merged.sections = []
    merged.blocks = []
    merged.paragraphs = []
    merged.tables = []
    merged.images = []
    merged.diagnostics = []

    for document in documents:
        merged.pages.extend(document.pages)
        merged.sections.extend(document.sections)
        merged.blocks.extend(document.blocks)
        merged.paragraphs.extend(document.paragraphs)
        merged.tables.extend(document.tables)
        merged.images.extend(document.images)
        merged.diagnostics.extend(document.diagnostics)

    merged.pages.sort(key=lambda page: page.page_number)
    merged.blocks.sort(key=lambda block: (block.page_number, block.reading_order_index or 0))
    merged.paragraphs.sort(key=lambda block: (block.page_number, block.reading_order_index or 0))
    merged.tables.sort(key=lambda table: (table.page_number or 0, table.reading_order_index or 0))
    merged.images.sort(key=lambda image: (image.page_number, image.reading_order_index or 0))
    merged.sections.sort(key=lambda section: (section.page_start, section.section_id))
    merged.metadata.page_count = len(merged.pages)

    if merged.parser_metadata is not None:
        first_metadata = documents[0].parser_metadata
        last_metadata = documents[-1].parser_metadata
        merged.parser_metadata = ParserMetadata(
            parser_name=merged.parser_metadata.parser_name,
            parser_version=merged.parser_metadata.parser_version,
            parse_status=merge_parse_statuses(documents),
            parse_strategy="split_pages",
            started_at=first_metadata.started_at if first_metadata is not None else None,
            completed_at=last_metadata.completed_at if last_metadata is not None else None,
            diagnostics=merged.diagnostics,
        )

    return merged


def process_document(document_path: Path, page_range: tuple[int, int] | None) -> ParsedDocument:
    return DoclingDocumentParser(page_range=page_range).parse(document_path)


def process_split_pages(document_path: Path, page_range: tuple[int, int]) -> ParsedDocument:
    parsed_pages = []
    start_page, end_page = page_range
    for page_no in range(start_page, end_page + 1):
        print(f"Processing page {page_no}...")
        page_document = process_document(document_path, (page_no, page_no))
        parsed_pages.append(prefix_document_ids_for_split_page(page_document, page_no))
    return merge_parsed_documents(parsed_pages)


def main() -> None:
    args = parse_args()
    print("Validating document...")
    document_path = validate_document_path(args.document_path)
    page_range = validate_page_range(args.page_range)
    validate_split_pages(args.split_pages, page_range)

    if args.split_pages:
        if page_range is None:
            raise ValueError("--split-pages requires --page-range START END.")
        output_dir = build_split_output_dir(document_path, page_range)
    else:
        output_dir = build_output_dir(document_path, page_range)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing with Docling adapter...")
    if args.split_pages:
        if page_range is None:
            raise ValueError("--split-pages requires --page-range START END.")
        parsed_document = process_split_pages(document_path, page_range)
    else:
        parsed_document = process_document(document_path, page_range)

    output_path = output_dir / "parsed_document.json"
    print("Writing normalized output...")
    write_json(output_path, parsed_document.model_dump(mode="json"))

    parse_status = (
        parsed_document.parser_metadata.parse_status
        if parsed_document.parser_metadata is not None
        else None
    )
    print(f"Processed: {document_path}")
    print(f"Output file: {output_path}")
    print(f"Parse status: {parse_status}")
    print(
        "Counts: "
        f"{len(parsed_document.pages)} pages, "
        f"{len(parsed_document.blocks)} blocks, "
        f"{len(parsed_document.tables)} tables, "
        f"{len(parsed_document.images)} images, "
        f"{len(parsed_document.sections)} sections, "
        f"{len(parsed_document.diagnostics)} diagnostics"
    )
    print("Done.")


if __name__ == "__main__":
    main()
