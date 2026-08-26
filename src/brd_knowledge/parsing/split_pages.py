from __future__ import annotations

import multiprocessing as mp
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from typing import Any

from brd_knowledge.parsing.docling_parser import DoclingDocumentParser
from brd_knowledge.parsing.section_reconstruction import (
    is_semantic_block,
    rebuild_document_sections,
)
from brd_knowledge.schemas.document import (
    DocumentBlock,
    DocumentMetadata,
    Page,
    ParsedDocument,
    ParsedImage,
    ParserDiagnostic,
    ParserMetadata,
    ParseStatus,
)
from brd_knowledge.schemas.requirement import ExtractedRequirement
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable, TableCell


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
        merged.blocks.extend(document.blocks)
        merged.paragraphs.extend(document.paragraphs)
        merged.tables.extend(document.tables)
        merged.images.extend(document.images)
        merged.diagnostics.extend(document.diagnostics)

    merged.pages.sort(key=lambda page: page.page_number)
    for page in merged.pages:
        page.blocks = sorted(
            (block for block in page.blocks if is_semantic_block(block)),
            key=lambda block: (
                block.reading_order_index
                if block.reading_order_index is not None
                else 10**9,
                block.block_id,
            ),
        )
    merged.blocks = [block for block in merged.blocks if is_semantic_block(block)]
    merged.paragraphs = [
        paragraph for paragraph in merged.paragraphs if is_semantic_block(paragraph)
    ]
    merged.blocks.sort(
        key=lambda block: (
            block.page_number,
            block.reading_order_index
            if block.reading_order_index is not None
            else 10**9,
            block.block_id,
        )
    )
    merged.paragraphs.sort(
        key=lambda block: (
            block.page_number,
            block.reading_order_index
            if block.reading_order_index is not None
            else 10**9,
            block.block_id,
        )
    )
    merged.tables.sort(
        key=lambda table: (
            table.page_number or 0,
            table.reading_order_index
            if table.reading_order_index is not None
            else 10**9,
            table.table_id,
        )
    )
    merged.images.sort(
        key=lambda image: (
            image.page_number,
            image.reading_order_index
            if image.reading_order_index is not None
            else 10**9,
            image.image_id,
        )
    )
    merged.sections = rebuild_document_sections(merged.metadata.document_id, merged.blocks)
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


def process_document_worker(
    document_path: str,
    page_number: int,
    result_queue: Any,
) -> None:
    try:
        parsed_document = process_document(Path(document_path), (page_number, page_number))
        result_queue.put(("success", parsed_document.model_dump(mode="json")))
    except Exception as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))


def process_document_with_timeout(
    document_path: Path,
    page_number: int,
    timeout_seconds: float | None,
) -> ParsedDocument:
    if timeout_seconds is None:
        return process_document(document_path, (page_number, page_number))

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=process_document_worker,
        args=(str(document_path), page_number, result_queue),
    )
    process.start()

    try:
        result = result_queue.get(timeout=timeout_seconds)
    except Empty as exc:
        process.terminate()
        process.join()
        raise TimeoutError(
            f"Page {page_number} exceeded timeout of {timeout_seconds} seconds."
        ) from exc

    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join()

    status = result[0]
    if status == "success":
        return ParsedDocument.model_validate(result[1])
    error_type = result[1]
    error_message = result[2]
    raise RuntimeError(f"{error_type}: {error_message}")


def build_failed_page_document(
    document_path: Path,
    page_number: int,
    exc: Exception,
) -> ParsedDocument:
    now = datetime.now(UTC)
    diagnostic = ParserDiagnostic(
        severity="error",
        parser_name="docling",
        page_number=page_number,
        error_type=type(exc).__name__,
        message=str(exc),
    )
    return ParsedDocument(
        metadata=DocumentMetadata(
            document_id=document_path.stem,
            filename=document_path.name,
            source_path=document_path,
            file_type=document_path.suffix.lower().lstrip("."),
            page_count=1,
            parser_name="docling",
        ),
        parser_metadata=ParserMetadata(
            parser_name="docling",
            parse_status="failed",
            parse_strategy="split_pages",
            started_at=now,
            completed_at=now,
            diagnostics=[diagnostic],
        ),
        pages=[
            Page(
                page_number=page_number,
                parse_status="failed",
                parser_name="docling",
                diagnostics=[diagnostic],
            )
        ],
        diagnostics=[diagnostic],
    )


def process_split_pages(
    document_path: Path,
    page_range: tuple[int, int],
    page_timeout_seconds: float | None = None,
) -> ParsedDocument:
    parsed_pages = []
    start_page, end_page = page_range
    for page_no in range(start_page, end_page + 1):
        print(f"Processing page {page_no}...")
        try:
            page_document = process_document_with_timeout(
                document_path,
                page_no,
                page_timeout_seconds,
            )
            parsed_pages.append(prefix_document_ids_for_split_page(page_document, page_no))
        except Exception as exc:
            print(f"Page {page_no} failed: {type(exc).__name__}: {exc}")
            parsed_pages.append(build_failed_page_document(document_path, page_no, exc))
    return merge_parsed_documents(parsed_pages)
