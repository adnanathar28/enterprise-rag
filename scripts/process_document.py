from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brd_knowledge.parsing import split_pages
from brd_knowledge.parsing.docling_parser import DoclingDocumentParser
from brd_knowledge.schemas.document import ParsedDocument
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
    parser.add_argument(
        "--page-timeout-seconds",
        type=float,
        help="Optional per-page timeout for --split-pages processing.",
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


def validate_page_timeout(page_timeout_seconds: float | None) -> float | None:
    if page_timeout_seconds is None:
        return None
    if page_timeout_seconds <= 0:
        raise ValueError("--page-timeout-seconds must be greater than 0.")
    return page_timeout_seconds


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


def duplicate_values(values: list[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    return {value: count for value, count in sorted(counts.items()) if count > 1}


def build_processing_summary(
    parsed_document: ParsedDocument,
    document_path: Path,
    page_range: tuple[int, int] | None,
    split_pages: bool,
    page_timeout_seconds: float | None,
    output_path: Path,
    summary_path: Path,
    report_path: Path | None = None,
    markdown_path: Path | None = None,
) -> dict[str, Any]:
    parse_status = (
        parsed_document.parser_metadata.parse_status
        if parsed_document.parser_metadata is not None
        else None
    )
    failed_pages = [
        {
            "page_number": page.page_number,
            "diagnostics": [diagnostic.model_dump(mode="json") for diagnostic in page.diagnostics],
        }
        for page in parsed_document.pages
        if page.parse_status == "failed"
    ]
    diagnostics = [diagnostic.model_dump(mode="json") for diagnostic in parsed_document.diagnostics]

    duplicate_ids = {
        "blocks": duplicate_values([block.block_id for block in parsed_document.blocks]),
        "tables": duplicate_values([table.table_id for table in parsed_document.tables]),
        "images": duplicate_values([image.image_id for image in parsed_document.images]),
        "sections": duplicate_values([section.section_id for section in parsed_document.sections]),
        "cells": duplicate_values(
            [cell.cell_id for table in parsed_document.tables for cell in table.cells]
        ),
    }
    missing_provenance = {
        "blocks": [
            block.block_id
            for block in parsed_document.blocks
            if block.source is None or block.source.page_number is None
        ],
        "tables": [
            table.table_id
            for table in parsed_document.tables
            if table.source is None or table.source.page_number is None
        ],
        "images": [
            image.image_id
            for image in parsed_document.images
            if image.source is None or image.source.page_number is None
        ],
        "cells": [
            cell.cell_id
            for table in parsed_document.tables
            for cell in table.cells
            if cell.source is None or cell.source.page_number is None
        ],
    }
    empty_pages = [
        page.page_number
        for page in parsed_document.pages
        if page.parse_status != "failed"
        and not page.blocks
        and not page.tables
        and not page.images
    ]

    return {
        "source": {
            "path": str(document_path),
            "filename": document_path.name,
            "suffix": document_path.suffix.lower(),
            "size_bytes": document_path.stat().st_size,
            "page_range": page_range,
        },
        "processing": {
            "split_pages": split_pages,
            "page_timeout_seconds": page_timeout_seconds,
            "parse_status": parse_status,
            "parser_name": parsed_document.metadata.parser_name,
            "parser_version": parsed_document.metadata.parser_version,
        },
        "counts": {
            "pages": len(parsed_document.pages),
            "blocks": len(parsed_document.blocks),
            "tables": len(parsed_document.tables),
            "table_cells": sum(len(table.cells) for table in parsed_document.tables),
            "images": len(parsed_document.images),
            "sections": len(parsed_document.sections),
            "diagnostics": len(parsed_document.diagnostics),
            "failed_pages": len(failed_pages),
            "empty_pages": len(empty_pages),
        },
        "failed_pages": failed_pages,
        "empty_pages": empty_pages,
        "diagnostics": diagnostics,
        "checks": {
            "duplicate_ids": duplicate_ids,
            "missing_provenance": missing_provenance,
            "has_duplicate_ids": any(
                duplicate_ids_for_type for duplicate_ids_for_type in duplicate_ids.values()
            ),
            "has_missing_provenance": any(
                missing_ids for missing_ids in missing_provenance.values()
            ),
        },
        "outputs": {
            "parsed_document": str(output_path),
            "processing_summary": str(summary_path),
            "normalized_report": str(report_path) if report_path is not None else None,
            "normalized_output_markdown": str(markdown_path) if markdown_path is not None else None,
        },
    }


def markdown_escape_cell(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())


def pad_markdown_cell(value: str, width: int) -> str:
    return value.ljust(width)


def table_to_markdown(table: ParsedTable) -> list[str]:
    if not table.rows:
        return []

    max_columns = 0
    rendered_rows: list[dict[int, TableCell]] = []
    for row in table.rows:
        cells_by_column: dict[int, TableCell] = {}
        for cell in row.cells:
            for offset in range(cell.column_span):
                cells_by_column[cell.column_index + offset] = cell
            max_columns = max(max_columns, cell.column_index + cell.column_span)
        rendered_rows.append(cells_by_column)

    if max_columns == 0:
        return []

    table_rows = {row.row_index: row for row in table.rows}
    row_positions = {row.row_index: index for index, row in enumerate(table.rows)}
    row_indexes = {row.row_index for row in table.rows}
    for cell in table.cells:
        for row_offset in range(1, cell.row_span):
            target_row_index = cell.row_index + row_offset
            if target_row_index not in row_indexes:
                continue
            target_row = table_rows[target_row_index]
            rendered_row = rendered_rows[row_positions[target_row.row_index]]
            for column_offset in range(cell.column_span):
                rendered_row.setdefault(cell.column_index + column_offset, cell)

    rendered_values = []
    column_widths = [3] * max_columns
    for cells_by_column in rendered_rows:
        values = [
            markdown_escape_cell(cells_by_column[column_index].text)
            if column_index in cells_by_column
            else ""
            for column_index in range(max_columns)
        ]
        rendered_values.append(values)
        for column_index, value in enumerate(values):
            column_widths[column_index] = max(column_widths[column_index], len(value))

    lines = []
    for row_index, values in enumerate(rendered_values):
        padded_values = [
            pad_markdown_cell(value, column_widths[column_index])
            for column_index, value in enumerate(values)
        ]
        lines.append(f"| {' | '.join(padded_values)} |")
        if row_index == 0:
            separators = ["-" * width for width in column_widths]
            lines.append(f"| {' | '.join(separators)} |")
    return lines


def build_normalized_output_markdown(parsed_document: ParsedDocument) -> str:
    lines = [
        f"# {parsed_document.metadata.filename}",
        "",
    ]

    for page in parsed_document.pages:
        lines.extend([f"<!-- Page {page.page_number} -->", "", f"# Page {page.page_number}", ""])
        if page.parse_status != "success":
            lines.append(f"> **Page parse status:** {page.parse_status}")
            for diagnostic in page.diagnostics:
                diagnostic_type = diagnostic.error_type or "Diagnostic"
                lines.append(f"> **{diagnostic_type}:** {diagnostic.message}")
            lines.append("")

        page_items: list[tuple[int, str, Any]] = []
        for block in page.blocks:
            page_items.append((block.reading_order_index or 0, "block", block))
        for table in page.tables:
            page_items.append((table.reading_order_index or 0, "table", table))
        for image in page.images:
            page_items.append((image.reading_order_index or 0, "image", image))

        for _, item_type, item in sorted(page_items, key=lambda entry: entry[0]):
            if item_type == "block":
                if item.block_type == "section_header":
                    lines.extend([f"## {item.text}", ""])
                elif item.block_type == "list_item":
                    lines.extend([f"- {item.text}", ""])
                else:
                    lines.extend([item.text, ""])
            elif item_type == "table":
                table_lines = table_to_markdown(item)
                if table_lines:
                    lines.extend(table_lines)
                    lines.append("")
            elif item_type == "image":
                lines.extend(["<!-- image -->", ""])

    return "\n".join(lines).rstrip() + "\n"


def build_normalized_report(summary: dict[str, Any], parsed_document: ParsedDocument) -> str:
    checks = summary["checks"]
    counts = summary["counts"]
    processing = summary["processing"]
    lines = [
        "Normalized Processing Report",
        "",
        f"Source: {summary['source']['path']}",
        f"Page range: {summary['source']['page_range']}",
        f"Split pages: {processing['split_pages']}",
        f"Page timeout seconds: {processing['page_timeout_seconds']}",
        f"Parse status: {processing['parse_status']}",
        "",
        "Counts",
        f"- Pages: {counts['pages']}",
        f"- Blocks: {counts['blocks']}",
        f"- Tables: {counts['tables']}",
        f"- Table cells: {counts['table_cells']}",
        f"- Images: {counts['images']}",
        f"- Sections: {counts['sections']}",
        f"- Diagnostics: {counts['diagnostics']}",
        f"- Failed pages: {counts['failed_pages']}",
        f"- Empty pages: {counts['empty_pages']}",
        "",
        "Checks",
        f"- Duplicate IDs: {checks['has_duplicate_ids']}",
        f"- Missing provenance: {checks['has_missing_provenance']}",
    ]

    if summary["failed_pages"]:
        lines.extend(["", "Failed Pages"])
        for failed_page in summary["failed_pages"]:
            diagnostics = failed_page["diagnostics"]
            messages = "; ".join(diagnostic["message"] for diagnostic in diagnostics)
            lines.append(f"- Page {failed_page['page_number']}: {messages}")

    if summary["empty_pages"]:
        lines.extend(["", "Empty Pages"])
        lines.append("- " + ", ".join(str(page_number) for page_number in summary["empty_pages"]))

    lines.extend(["", "Per-Page Summary"])
    for page in parsed_document.pages:
        lines.append(
            f"- Page {page.page_number}: {page.parse_status}, "
            f"blocks={len(page.blocks)}, tables={len(page.tables)}, images={len(page.images)}, "
            f"diagnostics={len(page.diagnostics)}"
        )
        for diagnostic in page.diagnostics:
            diagnostic_type = diagnostic.error_type or "Diagnostic"
            lines.append(f"  - {diagnostic_type}: {diagnostic.message}")

    if parsed_document.tables:
        lines.extend(["", "Tables"])
        for table in parsed_document.tables:
            lines.append(
                f"- {table.table_id}: page={table.page_number}, "
                f"rows={len(table.rows)}, cells={len(table.cells)}, "
                f"quality_notes={len(table.quality_notes)}"
            )

    return "\n".join(lines) + "\n"


def process_document(document_path: Path, page_range: tuple[int, int] | None) -> ParsedDocument:
    return DoclingDocumentParser(page_range=page_range).parse(document_path)


def main() -> None:
    args = parse_args()
    print("Validating document...")
    document_path = validate_document_path(args.document_path)
    page_range = validate_page_range(args.page_range)
    validate_split_pages(args.split_pages, page_range)
    page_timeout_seconds = validate_page_timeout(args.page_timeout_seconds)

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
        parsed_document = split_pages.process_split_pages(
            document_path,
            page_range,
            page_timeout_seconds,
        )
    else:
        parsed_document = process_document(document_path, page_range)

    output_path = output_dir / "parsed_document.json"
    summary_path = output_dir / "processing_summary.json"
    report_path = output_dir / "normalized_report.txt"
    markdown_path = output_dir / "normalized_output.md"
    print("Writing normalized output...")
    write_json(output_path, parsed_document.model_dump(mode="json"))
    summary = build_processing_summary(
        parsed_document=parsed_document,
        document_path=document_path,
        page_range=page_range,
        split_pages=args.split_pages,
        page_timeout_seconds=page_timeout_seconds,
        output_path=output_path,
        summary_path=summary_path,
        report_path=report_path,
        markdown_path=markdown_path,
    )
    write_json(summary_path, summary)
    report_path.write_text(build_normalized_report(summary, parsed_document), encoding="utf-8")
    markdown_path.write_text(build_normalized_output_markdown(parsed_document), encoding="utf-8")

    parse_status = (
        parsed_document.parser_metadata.parse_status
        if parsed_document.parser_metadata is not None
        else None
    )
    print(f"Processed: {document_path}")
    print(f"Output file: {output_path}")
    print(f"Summary file: {summary_path}")
    print(f"Report file: {report_path}")
    print(f"Markdown file: {markdown_path}")
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
