from brd_knowledge.extraction.schemas import (
    ExtractionContentType,
    ExtractionSourceRef,
    ExtractionUnit,
)
from brd_knowledge.schemas.document import DocumentBlock, ParsedDocument
from brd_knowledge.schemas.table import ParsedTable, TableRow


class RequirementExtractionInputBuilder:
    def build(self, parsed_document: ParsedDocument) -> list[ExtractionUnit]:
        document_id = parsed_document.metadata.document_id
        section_by_block_id = self._section_lookup(parsed_document)
        units = []

        for block in parsed_document.blocks:
            unit = self._unit_from_block(
                block=block,
                document_id=document_id,
                section_by_block_id=section_by_block_id,
            )
            if unit is not None:
                units.append(unit)

        for table in parsed_document.tables:
            units.extend(self._units_from_table(table, document_id))

        return sorted(
            units,
            key=lambda unit: (
                unit.page_number,
                unit.reading_order_index if unit.reading_order_index is not None else 0,
                unit.unit_id,
            ),
        )

    def _unit_from_block(
        self,
        block: DocumentBlock,
        document_id: str,
        section_by_block_id: dict[str, tuple[str, str]],
    ) -> ExtractionUnit | None:
        text = " ".join(block.text.split())
        if not text:
            return None
        section = section_by_block_id.get(block.block_id)
        section_id = section[0] if section is not None else None
        section_title = section[1] if section is not None else None
        content_type = self._block_content_type(block.block_type)
        return ExtractionUnit(
            unit_id=f"unit-{block.block_id}",
            document_id=document_id,
            page_number=block.page_number,
            content_type=content_type,
            text=text,
            source_refs=[
                ExtractionSourceRef(
                    document_id=document_id,
                    page_number=block.page_number,
                    section_id=section_id,
                    block_id=block.block_id,
                    reading_order_index=block.reading_order_index,
                )
            ],
            section_id=section_id,
            section_title=section_title,
            reading_order_index=block.reading_order_index,
        )

    def _units_from_table(
        self,
        table: ParsedTable,
        document_id: str,
    ) -> list[ExtractionUnit]:
        units = []
        headers = self._header_cells_by_column(table)
        for row in table.rows:
            if self._is_header_row(row):
                continue
            text = self._row_text(row, headers)
            if not text:
                continue
            page_number = table.page_number
            if page_number is None:
                continue
            cell_ids = [
                cell.cell_id
                for cell in sorted(row.cells, key=lambda cell: cell.column_index)
                if cell.cell_id is not None
            ]
            units.append(
                ExtractionUnit(
                    unit_id=f"unit-{table.table_id}-row-{row.row_index}",
                    document_id=document_id,
                    page_number=page_number,
                    content_type="table_row",
                    text=text,
                    source_refs=[
                        ExtractionSourceRef(
                            document_id=document_id,
                            page_number=page_number,
                            table_id=table.table_id,
                            cell_ids=cell_ids,
                            reading_order_index=table.reading_order_index,
                        )
                    ],
                    reading_order_index=table.reading_order_index,
                    quality_notes=table.quality_notes,
                )
            )
        return units

    def _section_lookup(self, parsed_document: ParsedDocument) -> dict[str, tuple[str, str]]:
        section_by_block_id = {}
        for section in parsed_document.sections:
            for block in section.blocks:
                section_by_block_id[block.block_id] = (section.section_id, section.title)
            for paragraph in section.paragraphs:
                section_by_block_id[paragraph.block_id] = (section.section_id, section.title)
            if section.heading_block is not None:
                section_by_block_id[section.heading_block.block_id] = (
                    section.section_id,
                    section.title,
                )
        return section_by_block_id

    def _block_content_type(self, block_type: str) -> ExtractionContentType:
        if block_type == "list_item":
            return "list_item"
        if block_type == "section_header":
            return "section_header"
        return "paragraph"

    def _header_cells_by_column(self, table: ParsedTable) -> dict[int, str]:
        headers = {}
        for cell in table.cells:
            if not cell.is_header:
                continue
            text = " ".join(cell.text.split())
            if text:
                headers[cell.column_index] = text
        return headers

    def _is_header_row(self, row: TableRow) -> bool:
        return bool(row.cells) and all(cell.is_header for cell in row.cells)

    def _row_text(self, row: TableRow, headers: dict[int, str]) -> str:
        parts = []
        for cell in sorted(row.cells, key=lambda item: item.column_index):
            text = " ".join(cell.text.split())
            if not text:
                continue
            header = headers.get(cell.column_index)
            parts.append(f"{header}: {text}" if header else text)
        return " | ".join(parts)


def build_requirement_extraction_units(parsed_document: ParsedDocument) -> list[ExtractionUnit]:
    return RequirementExtractionInputBuilder().build(parsed_document)
