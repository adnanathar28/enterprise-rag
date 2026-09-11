from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

from brd_knowledge.schemas.chunk import Chunk, ChunkContentType
from brd_knowledge.schemas.document import DocumentBlock, ParsedDocument
from brd_knowledge.schemas.section import DocumentSection
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable, TableCell

ChunkableItem = DocumentBlock | ParsedTable


class StructureAwareChunker:
    def __init__(self, max_characters: int = 3000) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero.")
        self.max_characters = max_characters

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        assigned_block_ids: set[str] = set()
        assigned_table_ids: set[str] = set()

        for section, path in self._walk_sections(document.sections):
            chunks.extend(self._chunk_section(document.metadata.document_id, section, path))
            assigned_block_ids.update(block.block_id for block in section.blocks)
            if section.heading_block is not None:
                assigned_block_ids.add(section.heading_block.block_id)
            assigned_table_ids.update(table.table_id for table in section.tables)

        unsectioned_items: list[ChunkableItem] = [
            block for block in document.blocks if block.block_id not in assigned_block_ids
        ]
        unsectioned_items.extend(
            table for table in document.tables if table.table_id not in assigned_table_ids
        )
        chunks.extend(
            self._chunk_items(
                document_id=document.metadata.document_id,
                section_id=None,
                section_title=None,
                section_path=[],
                items=self._ordered_items(unsectioned_items),
                heading_block=None,
            )
        )
        return chunks

    def _walk_sections(
        self,
        sections: Sequence[DocumentSection],
        parent_path: Sequence[str] = (),
    ) -> Iterable[tuple[DocumentSection, list[str]]]:
        for section in sections:
            path = [*parent_path, section.title]
            yield section, path
            yield from self._walk_sections(section.child_sections, path)

    def _chunk_section(
        self,
        document_id: str,
        section: DocumentSection,
        section_path: list[str],
    ) -> list[Chunk]:
        heading_id = section.heading_block.block_id if section.heading_block is not None else None
        content_blocks = [block for block in section.blocks if block.block_id != heading_id]
        items: list[ChunkableItem] = [*content_blocks, *section.tables]
        return self._chunk_items(
            document_id=document_id,
            section_id=section.section_id,
            section_title=section.title,
            section_path=section_path,
            items=self._ordered_items(items),
            heading_block=section.heading_block,
        )

    def _chunk_items(
        self,
        document_id: str,
        section_id: str | None,
        section_title: str | None,
        section_path: list[str],
        items: Sequence[ChunkableItem],
        heading_block: DocumentBlock | None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        prose_blocks: list[DocumentBlock] = []
        heading_pending = heading_block

        def flush_prose() -> None:
            nonlocal heading_pending
            if not prose_blocks:
                return
            prose_chunks = self._build_prose_chunks(
                document_id=document_id,
                section_id=section_id,
                section_title=section_title,
                section_path=section_path,
                blocks=prose_blocks,
                heading_block=heading_pending,
                starting_sequence=len(chunks),
            )
            chunks.extend(prose_chunks)
            prose_blocks.clear()
            if prose_chunks:
                heading_pending = None

        for item in items:
            if isinstance(item, ParsedTable):
                flush_prose()
                chunks.append(
                    self._build_table_chunk(
                        document_id=document_id,
                        section_id=section_id,
                        section_title=section_title,
                        section_path=section_path,
                        table=item,
                        heading_block=heading_pending,
                        sequence=len(chunks),
                    )
                )
                heading_pending = None
            else:
                prose_blocks.append(item)
        flush_prose()

        if not chunks and heading_pending is not None:
            chunks.extend(
                self._build_prose_chunks(
                    document_id=document_id,
                    section_id=section_id,
                    section_title=section_title,
                    section_path=section_path,
                    blocks=[],
                    heading_block=heading_pending,
                    starting_sequence=0,
                )
            )
        return chunks

    def _build_prose_chunks(
        self,
        document_id: str,
        section_id: str | None,
        section_title: str | None,
        section_path: list[str],
        blocks: Sequence[DocumentBlock],
        heading_block: DocumentBlock | None,
        starting_sequence: int,
    ) -> list[Chunk]:
        contextual_prefix = self._context_prefix(section_path)
        content_limit = max(1, self.max_characters - len(contextual_prefix))
        segments: list[tuple[str, DocumentBlock]] = []
        for block in blocks:
            segments.extend((text, block) for text in self._split_text(block.text, content_limit))

        if not segments and heading_block is not None:
            segments = [("" if contextual_prefix else heading_block.text, heading_block)]

        groups: list[list[tuple[str, DocumentBlock]]] = []
        current: list[tuple[str, DocumentBlock]] = []
        current_length = len(contextual_prefix)
        for segment in segments:
            separator_length = 2 if current else 0
            proposed_length = current_length + separator_length + len(segment[0])
            if current and proposed_length > self.max_characters:
                groups.append(current)
                current = []
                current_length = len(contextual_prefix)
            current.append(segment)
            current_length += (2 if len(current) > 1 else 0) + len(segment[0])
        if current:
            groups.append(current)

        chunks = []
        for offset, group in enumerate(groups):
            is_first = offset == 0
            group_blocks = self._unique_blocks(block for _, block in group)
            source_blocks = list(group_blocks)
            if is_first and heading_block is not None:
                source_blocks = self._unique_blocks([heading_block, *source_blocks])
            body = "\n\n".join(text for text, _ in group)
            text = (f"{contextual_prefix}{body}" if contextual_prefix else body).rstrip()
            chunks.append(
                self._make_chunk(
                    document_id=document_id,
                    section_id=section_id,
                    section_title=section_title,
                    section_path=section_path,
                    content_type="prose",
                    text=text,
                    pages=[block.page_number for block in source_blocks],
                    source_blocks=source_blocks,
                    source_tables=[],
                    quality_notes=[],
                    sequence=starting_sequence + offset,
                )
            )
        return chunks

    def _build_table_chunk(
        self,
        document_id: str,
        section_id: str | None,
        section_title: str | None,
        section_path: list[str],
        table: ParsedTable,
        heading_block: DocumentBlock | None,
        sequence: int,
    ) -> Chunk:
        use_native_fallback = (
            table.native_text_coverage is not None
            and table.native_text_coverage < 0.85
            and bool(table.native_text)
        )
        if use_native_fallback:
            body = table.native_text or ""
        else:
            body = self._render_structured_table(table)
        prefix = self._context_prefix(section_path)
        text = f"{prefix}{body}" if prefix else body
        source_blocks = [heading_block] if heading_block is not None else []
        pages = list(table.page_numbers)
        if table.page_number is not None:
            pages.append(table.page_number)
        pages.extend(block.page_number for block in source_blocks)
        return self._make_chunk(
            document_id=document_id,
            section_id=section_id,
            section_title=section_title,
            section_path=section_path,
            content_type="table",
            text=text,
            pages=pages,
            source_blocks=source_blocks,
            source_tables=[table],
            quality_notes=list(table.quality_notes),
            sequence=sequence,
        )

    def _make_chunk(
        self,
        document_id: str,
        section_id: str | None,
        section_title: str | None,
        section_path: list[str],
        content_type: ChunkContentType,
        text: str,
        pages: Sequence[int],
        source_blocks: Sequence[DocumentBlock],
        source_tables: Sequence[ParsedTable],
        quality_notes: list[str],
        sequence: int,
    ) -> Chunk:
        source_block_ids = [block.block_id for block in source_blocks]
        source_table_ids = [table.table_id for table in source_tables]
        provenance = self._unique_sources(
            [block.source for block in source_blocks]
            + [table.source for table in source_tables]
        )
        chunk_id = self._chunk_id(
            document_id=document_id,
            section_id=section_id,
            content_type=content_type,
            sequence=sequence,
            source_block_ids=source_block_ids,
            source_table_ids=source_table_ids,
            text=text,
        )
        valid_pages = list(pages)
        if not valid_pages:
            raise ValueError("Cannot create a chunk without page provenance.")
        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            section_id=section_id,
            section_title=section_title,
            section_path=list(section_path),
            content_type=content_type,
            text=text,
            page_start=min(valid_pages),
            page_end=max(valid_pages),
            source_block_ids=source_block_ids,
            source_table_ids=source_table_ids,
            provenance=provenance,
            quality_notes=quality_notes,
        )

    def _split_text(self, text: str, limit: int) -> list[str]:
        if len(text) <= limit:
            return [text]
        return [text[start : start + limit] for start in range(0, len(text), limit)]

    def _render_structured_table(self, table: ParsedTable) -> str:
        ordered_rows = sorted(table.rows, key=lambda item: item.row_index)
        if not ordered_rows:
            return ""

        cells_by_row: dict[int, dict[int, TableCell]] = {
            row.row_index: {} for row in ordered_rows
        }
        max_columns = 0
        for row in ordered_rows:
            for cell in sorted(row.cells, key=lambda item: item.column_index):
                max_columns = max(max_columns, cell.column_index + cell.column_span)
                for column_offset in range(cell.column_span):
                    cells_by_row[row.row_index][cell.column_index + column_offset] = cell

        row_indexes = set(cells_by_row)
        for row in ordered_rows:
            for cell in row.cells:
                for row_offset in range(1, cell.row_span):
                    target_row_index = cell.row_index + row_offset
                    if target_row_index not in row_indexes:
                        continue
                    for column_offset in range(cell.column_span):
                        cells_by_row[target_row_index].setdefault(
                            cell.column_index + column_offset,
                            cell,
                        )

        lines = []
        for row in ordered_rows:
            cells = cells_by_row[row.row_index]
            lines.append(
                " | ".join(
                    " ".join(cells[column_index].text.split())
                    if column_index in cells
                    else ""
                    for column_index in range(max_columns)
                )
            )
        return "\n".join(lines)

    def _context_prefix(self, section_path: Sequence[str]) -> str:
        if not section_path:
            return ""
        return f"Section: {' > '.join(section_path)}\n\n"

    def _ordered_items(self, items: Sequence[ChunkableItem]) -> list[ChunkableItem]:
        return sorted(items, key=self._order_key)

    def _order_key(self, item: ChunkableItem) -> tuple[int, int, int, str]:
        if isinstance(item, ParsedTable):
            page_number = item.page_number or 10**9
            type_order = 1
            item_id = item.table_id
        else:
            page_number = item.page_number
            type_order = 0
            item_id = item.block_id
        return (
            page_number,
            item.reading_order_index if item.reading_order_index is not None else 10**9,
            type_order,
            item_id,
        )

    def _unique_blocks(self, blocks: Iterable[DocumentBlock]) -> list[DocumentBlock]:
        unique: dict[str, DocumentBlock] = {}
        for block in blocks:
            unique.setdefault(block.block_id, block)
        return list(unique.values())

    def _unique_sources(
        self,
        sources: Sequence[SourceReference | None],
    ) -> list[SourceReference]:
        unique: dict[str, SourceReference] = {}
        for source in sources:
            if source is None:
                continue
            key = json.dumps(source.model_dump(mode="json"), sort_keys=True)
            unique.setdefault(key, source)
        return list(unique.values())

    def _chunk_id(
        self,
        document_id: str,
        section_id: str | None,
        content_type: ChunkContentType,
        sequence: int,
        source_block_ids: Sequence[str],
        source_table_ids: Sequence[str],
        text: str,
    ) -> str:
        identity = json.dumps(
            {
                "document_id": document_id,
                "section_id": section_id,
                "content_type": content_type,
                "sequence": sequence,
                "source_block_ids": list(source_block_ids),
                "source_table_ids": list(source_table_ids),
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        return f"chunk-{digest}"
