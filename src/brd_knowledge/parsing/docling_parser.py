from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, cast

from brd_knowledge.core.exceptions import ParserError
from brd_knowledge.parsing.base import DocumentParser
from brd_knowledge.schemas.document import (
    DocumentBlock,
    DocumentMetadata,
    Page,
    ParsedDocument,
    ParsedImage,
    ParserDiagnostic,
    ParserMetadata,
    ParseStatus,
    TextBlock,
)
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.source import BoundingBox, CoordinateOrigin, SourceReference
from brd_knowledge.schemas.table import ParsedTable, TableCell, TableRow


class DoclingDocumentParser(DocumentParser):
    parser_name = "docling"

    def __init__(self, page_range: tuple[int, int] | None = None) -> None:
        self.page_range = page_range

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".pdf", ".docx"}

    def parse(self, file_path: Path) -> ParsedDocument:
        resolved_path = file_path.resolve()
        if not resolved_path.exists():
            raise ParserError(f"Document does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise ParserError(f"Document path is not a file: {resolved_path}")
        if not self.can_parse(resolved_path):
            raise ParserError(f"Unsupported file type for Docling parser: {resolved_path.suffix}")

        try:
            document_converter_module = import_module("docling.document_converter")
            document_converter_class = cast(Any, document_converter_module).DocumentConverter
        except ImportError as exc:
            raise ParserError("Docling is not installed or cannot be imported.") from exc

        started_at = datetime.now(UTC)
        convert_kwargs: dict[str, Any] = {}
        if self.page_range is not None:
            convert_kwargs["page_range"] = self.page_range

        try:
            result = document_converter_class().convert(
                resolved_path,
                raises_on_error=False,
                **convert_kwargs,
            )
        except Exception as exc:
            raise ParserError(f"Docling failed to convert document: {resolved_path}") from exc

        completed_at = datetime.now(UTC)
        docling_version = metadata.version("docling")
        docling_document = result.document
        document_id = resolved_path.stem
        parse_status = self._map_parse_status(getattr(result, "status", None))
        diagnostics = self._build_diagnostics(result, docling_version)
        reading_order = self._build_reading_order(docling_document)

        blocks = self._build_blocks(docling_document, document_id, reading_order)
        tables = self._build_tables(docling_document, document_id, reading_order)
        images = self._build_images(docling_document, document_id, reading_order)
        document_blocks: list[DocumentBlock] = list(blocks)
        pages = self._build_pages(
            docling_document,
            parse_status,
            diagnostics,
            document_blocks,
            tables,
            images,
        )
        sections = self._build_sections(document_id, blocks)

        return ParsedDocument(
            metadata=DocumentMetadata(
                document_id=document_id,
                filename=resolved_path.name,
                source_path=resolved_path,
                file_type=resolved_path.suffix.lower().lstrip("."),
                page_count=len(getattr(docling_document, "pages", {}) or {}),
                parser_name=self.parser_name,
                parser_version=docling_version,
            ),
            parser_metadata=ParserMetadata(
                parser_name=self.parser_name,
                parser_version=docling_version,
                parse_status=parse_status,
                parse_strategy="page_range" if self.page_range else "single_conversion",
                started_at=started_at,
                completed_at=completed_at,
                diagnostics=diagnostics,
            ),
            pages=pages,
            sections=sections,
            blocks=document_blocks,
            paragraphs=[block for block in blocks if isinstance(block, TextBlock)],
            tables=tables,
            images=images,
            diagnostics=diagnostics,
        )

    def _build_reading_order(self, docling_document: Any) -> dict[str, int]:
        reading_order = {}
        for index, item_tuple in enumerate(docling_document.iterate_items(with_groups=True)):
            item = item_tuple[0]
            self_ref = getattr(item, "self_ref", None)
            if self_ref is not None:
                reading_order[str(self_ref)] = index
        return reading_order

    def _build_blocks(
        self,
        docling_document: Any,
        document_id: str,
        reading_order: dict[str, int],
    ) -> list[TextBlock]:
        blocks = []
        for index, item in enumerate(getattr(docling_document, "texts", []) or []):
            text = getattr(item, "text", None)
            if not isinstance(text, str) or not text.strip():
                continue

            self_ref = self._self_ref(item)
            page_number = self._first_page_number(item)
            if page_number is None:
                continue

            reading_order_index = reading_order.get(self_ref, index)
            block_id = self._id_from_ref(self_ref, fallback=f"block-{index}")
            bounding_box = self._first_bounding_box(item)
            blocks.append(
                TextBlock(
                    block_id=block_id,
                    page_number=page_number,
                    text=text,
                    block_type=self._label(item),
                    reading_order_index=reading_order_index,
                    bounding_box=bounding_box,
                    source=SourceReference(
                        document_id=document_id,
                        page_number=page_number,
                        parser_item_id=self_ref,
                        block_id=block_id,
                        reading_order_index=reading_order_index,
                        text_excerpt=self._excerpt(text),
                        bounding_box=bounding_box,
                    ),
                )
            )
        return sorted(blocks, key=lambda block: (block.page_number, block.reading_order_index or 0))

    def _build_tables(
        self,
        docling_document: Any,
        document_id: str,
        reading_order: dict[str, int],
    ) -> list[ParsedTable]:
        tables = []
        for index, table_item in enumerate(getattr(docling_document, "tables", []) or []):
            self_ref = self._self_ref(table_item)
            table_id = self._id_from_ref(self_ref, fallback=f"table-{index}")
            page_number = self._first_page_number(table_item)
            page_numbers = self._page_numbers(table_item)
            reading_order_index = reading_order.get(self_ref, index)
            bounding_box = self._first_bounding_box(table_item)
            rows, cells = self._build_table_rows_and_cells(
                table_item=table_item,
                document_id=document_id,
                table_id=table_id,
                page_number=page_number,
            )
            tables.append(
                ParsedTable(
                    table_id=table_id,
                    page_number=page_number,
                    page_numbers=page_numbers,
                    reading_order_index=reading_order_index,
                    rows=rows,
                    cells=cells,
                    caption=self._caption(table_item),
                    bounding_box=bounding_box,
                    source=SourceReference(
                        document_id=document_id,
                        page_number=page_number,
                        parser_item_id=self_ref,
                        table_id=table_id,
                        reading_order_index=reading_order_index,
                        bounding_box=bounding_box,
                    ),
                    quality_notes=self._table_quality_notes(table_item),
                )
            )
        return sorted(
            tables,
            key=lambda table: (table.page_number or 0, table.reading_order_index or 0),
        )

    def _build_table_rows_and_cells(
        self,
        table_item: Any,
        document_id: str,
        table_id: str,
        page_number: int | None,
    ) -> tuple[list[TableRow], list[TableCell]]:
        table_data = getattr(table_item, "data", None)
        grid = getattr(table_data, "grid", None) if table_data is not None else None
        raw_cells = getattr(table_data, "table_cells", []) if table_data is not None else []
        rows: list[TableRow] = []
        cells: list[TableCell] = []

        if grid:
            seen_cell_ids = set()
            for row_index, row_cells in enumerate(grid):
                row = TableRow(row_index=row_index)
                for fallback_column_index, raw_cell in enumerate(row_cells):
                    cell = self._build_table_cell(
                        raw_cell=raw_cell,
                        document_id=document_id,
                        table_id=table_id,
                        page_number=page_number,
                        fallback_row_index=row_index,
                        fallback_column_index=fallback_column_index,
                    )
                    row.cells.append(cell)
                    if cell.cell_id not in seen_cell_ids:
                        cells.append(cell)
                        seen_cell_ids.add(cell.cell_id)
                rows.append(row)
            return rows, cells

        for raw_cell in raw_cells:
            cell = self._build_table_cell(
                raw_cell=raw_cell,
                document_id=document_id,
                table_id=table_id,
                page_number=page_number,
                fallback_row_index=0,
                fallback_column_index=len(cells),
            )
            cells.append(cell)

        grouped_rows: dict[int, list[TableCell]] = defaultdict(list)
        for cell in cells:
            grouped_rows[cell.row_index].append(cell)
        rows = [
            TableRow(
                row_index=row_index,
                cells=sorted(row_cells, key=lambda cell: cell.column_index),
            )
            for row_index, row_cells in sorted(grouped_rows.items())
        ]
        return rows, cells

    def _build_table_cell(
        self,
        raw_cell: Any,
        document_id: str,
        table_id: str,
        page_number: int | None,
        fallback_row_index: int,
        fallback_column_index: int,
    ) -> TableCell:
        row_index = int(getattr(raw_cell, "start_row_offset_idx", fallback_row_index) or 0)
        column_index = int(getattr(raw_cell, "start_col_offset_idx", fallback_column_index) or 0)
        row_span = int(getattr(raw_cell, "row_span", 1) or 1)
        column_span = int(getattr(raw_cell, "col_span", 1) or 1)
        cell_id = f"{table_id}-r{row_index}-c{column_index}"
        bounding_box = self._bbox_from_raw(getattr(raw_cell, "bbox", None), page_number)
        text = getattr(raw_cell, "text", "") or ""
        return TableCell(
            cell_id=cell_id,
            row_index=row_index,
            column_index=column_index,
            text=text,
            row_span=row_span,
            column_span=column_span,
            bounding_box=bounding_box,
            source=SourceReference(
                document_id=document_id,
                page_number=page_number,
                table_id=table_id,
                cell_id=cell_id,
                text_excerpt=self._excerpt(text),
                bounding_box=bounding_box,
            ),
            is_header=bool(
                getattr(raw_cell, "column_header", False) or getattr(raw_cell, "row_header", False)
            ),
        )

    def _build_images(
        self,
        docling_document: Any,
        document_id: str,
        reading_order: dict[str, int],
    ) -> list[ParsedImage]:
        images = []
        for index, picture_item in enumerate(getattr(docling_document, "pictures", []) or []):
            self_ref = self._self_ref(picture_item)
            image_id = self._id_from_ref(self_ref, fallback=f"image-{index}")
            page_number = self._first_page_number(picture_item)
            if page_number is None:
                continue

            reading_order_index = reading_order.get(self_ref, index)
            bounding_box = self._first_bounding_box(picture_item)
            images.append(
                ParsedImage(
                    image_id=image_id,
                    page_number=page_number,
                    reading_order_index=reading_order_index,
                    bounding_box=bounding_box,
                    source=SourceReference(
                        document_id=document_id,
                        page_number=page_number,
                        parser_item_id=self_ref,
                        reading_order_index=reading_order_index,
                        bounding_box=bounding_box,
                    ),
                )
            )
        return sorted(images, key=lambda image: (image.page_number, image.reading_order_index or 0))

    def _build_pages(
        self,
        docling_document: Any,
        parse_status: ParseStatus,
        diagnostics: list[ParserDiagnostic],
        blocks: list[DocumentBlock],
        tables: list[ParsedTable],
        images: list[ParsedImage],
    ) -> list[Page]:
        pages = []
        blocks_by_page = self._group_by_page(blocks)
        tables_by_page = self._group_by_optional_page(tables)
        images_by_page = self._group_by_page(images)
        diagnostics_by_page: dict[int, list[ParserDiagnostic]] = defaultdict(list)
        for diagnostic in diagnostics:
            if diagnostic.page_number is not None:
                diagnostics_by_page[diagnostic.page_number].append(diagnostic)

        for page_no, page_item in sorted((getattr(docling_document, "pages", {}) or {}).items()):
            page_status = "failed" if diagnostics_by_page.get(page_no) else parse_status
            size = getattr(page_item, "size", None)
            pages.append(
                Page(
                    page_number=page_no,
                    width=getattr(size, "width", None),
                    height=getattr(size, "height", None),
                    parse_status=page_status,
                    parser_name=self.parser_name,
                    parser_version=metadata.version("docling"),
                    diagnostics=diagnostics_by_page.get(page_no, []),
                    blocks=blocks_by_page.get(page_no, []),
                    tables=tables_by_page.get(page_no, []),
                    images=images_by_page.get(page_no, []),
                )
            )
        return pages

    def _build_sections(self, document_id: str, blocks: list[TextBlock]) -> list[DocumentSection]:
        sections = []
        section_headers = [block for block in blocks if block.block_type == "section_header"]
        for index, heading in enumerate(section_headers):
            next_heading = section_headers[index + 1] if index + 1 < len(section_headers) else None
            section_blocks = [
                block
                for block in blocks
                if block.reading_order_index is not None
                and heading.reading_order_index is not None
                and block.reading_order_index >= heading.reading_order_index
                and (
                    next_heading is None
                    or next_heading.reading_order_index is None
                    or block.reading_order_index < next_heading.reading_order_index
                )
            ]
            document_blocks: list[DocumentBlock] = list(section_blocks)
            section_id = f"section-{index}"
            sections.append(
                DocumentSection(
                    section_id=section_id,
                    title=heading.text,
                    level=1,
                    page_start=heading.page_number,
                    page_end=(
                        section_blocks[-1].page_number if section_blocks else heading.page_number
                    ),
                    source=SourceReference(
                        document_id=document_id,
                        page_number=heading.page_number,
                        section_id=section_id,
                        block_id=heading.block_id,
                        reading_order_index=heading.reading_order_index,
                        text_excerpt=self._excerpt(heading.text),
                        bounding_box=heading.bounding_box,
                    ),
                    heading_block=heading,
                    blocks=document_blocks,
                    paragraphs=section_blocks,
                    tables=[],
                )
            )
        return sections

    def _build_diagnostics(self, result: Any, docling_version: str) -> list[ParserDiagnostic]:
        diagnostics = []
        for error in getattr(result, "errors", []) or []:
            diagnostics.append(
                ParserDiagnostic(
                    severity="error",
                    parser_name=self.parser_name,
                    parser_version=docling_version,
                    stage=getattr(error, "module_name", None),
                    page_number=getattr(error, "page_no", None),
                    error_type=getattr(error, "component_type", None),
                    message=getattr(error, "error_message", str(error)),
                )
            )
        return diagnostics

    def _first_bounding_box(self, item: Any) -> BoundingBox | None:
        prov = getattr(item, "prov", []) or []
        if not prov:
            return None
        return self._bbox_from_raw(
            getattr(prov[0], "bbox", None),
            getattr(prov[0], "page_no", None),
        )

    def _bbox_from_raw(self, raw_bbox: Any, page_number: int | None) -> BoundingBox | None:
        if raw_bbox is None or page_number is None:
            return None
        return BoundingBox(
            page_number=page_number,
            x0=float(getattr(raw_bbox, "l", 0) or 0),
            y0=float(getattr(raw_bbox, "t", 0) or 0),
            x1=float(getattr(raw_bbox, "r", 0) or 0),
            y1=float(getattr(raw_bbox, "b", 0) or 0),
            coordinate_origin=self._coordinate_origin(getattr(raw_bbox, "coord_origin", None)),
        )

    def _first_page_number(self, item: Any) -> int | None:
        page_numbers = self._page_numbers(item)
        return page_numbers[0] if page_numbers else None

    def _page_numbers(self, item: Any) -> list[int]:
        page_numbers = []
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                page_numbers.append(page_no)
        return sorted(set(page_numbers))

    def _label(self, item: Any) -> str:
        label = getattr(item, "label", None)
        return str(getattr(label, "value", label or type(item).__name__))

    def _self_ref(self, item: Any) -> str:
        return str(getattr(item, "self_ref", ""))

    def _id_from_ref(self, self_ref: str, fallback: str) -> str:
        if not self_ref:
            return fallback
        return self_ref.strip("#/").replace("/", "-") or fallback

    def _caption(self, table_item: Any) -> str | None:
        captions = getattr(table_item, "captions", []) or []
        caption_text = " ".join(
            caption.text for caption in captions if isinstance(getattr(caption, "text", None), str)
        ).strip()
        return caption_text or None

    def _table_quality_notes(self, table_item: Any) -> list[str]:
        notes = []
        data = getattr(table_item, "data", None)
        if data is None:
            notes.append("Table item has no table data.")
            return notes
        if not getattr(data, "grid", None):
            notes.append("Table item has no row grid.")
        return notes

    def _coordinate_origin(self, coord_origin: Any) -> CoordinateOrigin:
        value = str(coord_origin or "").lower()
        if value == "top-left" or value == "topleft":
            return "top_left"
        if value == "bottom-left" or value == "bottomleft":
            return "bottom_left"
        return "unknown"

    def _map_parse_status(self, status: Any) -> ParseStatus:
        value = str(getattr(status, "value", status)).lower()
        if value == "success":
            return "success"
        if value == "partial_success":
            return "partial_success"
        if value == "failure":
            return "failed"
        if value == "skipped":
            return "skipped"
        return "not_started"

    def _excerpt(self, text: str, limit: int = 240) -> str:
        compact_text = " ".join(text.split())
        if len(compact_text) <= limit:
            return compact_text
        return f"{compact_text[: limit - 3]}..."

    def _group_by_page(self, items: list[Any]) -> dict[int, list[Any]]:
        grouped_items: dict[int, list[Any]] = defaultdict(list)
        for item in items:
            grouped_items[item.page_number].append(item)
        return grouped_items

    def _group_by_optional_page(self, items: list[ParsedTable]) -> dict[int, list[ParsedTable]]:
        grouped_items: dict[int, list[ParsedTable]] = defaultdict(list)
        for item in items:
            if item.page_number is not None:
                grouped_items[item.page_number].append(item)
        return grouped_items
