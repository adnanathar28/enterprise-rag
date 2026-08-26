from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from brd_knowledge.schemas.document import DocumentBlock, ParsedImage
from brd_knowledge.schemas.table import ParsedTable

ReadingOrderItem = DocumentBlock | ParsedTable | ParsedImage


def stabilize_reading_order(
    blocks: Sequence[DocumentBlock],
    tables: Sequence[ParsedTable],
    images: Sequence[ParsedImage],
) -> None:
    """Assign unique page-local normalized order without changing source provenance."""
    items_by_page: dict[int, list[ReadingOrderItem]] = defaultdict(list)
    for block in blocks:
        items_by_page[block.page_number].append(block)
    for table in tables:
        if table.page_number is not None:
            items_by_page[table.page_number].append(table)
    for image in images:
        items_by_page[image.page_number].append(image)

    for items in items_by_page.values():
        items.sort(key=_source_order_key)
        for normalized_index, item in enumerate(items):
            item.reading_order_index = normalized_index


def _source_order_key(item: ReadingOrderItem) -> tuple[int, int, str]:
    source_index = item.reading_order_index
    if isinstance(item, DocumentBlock):
        item_type_order = 0
        item_id = item.block_id
    elif isinstance(item, ParsedTable):
        item_type_order = 1
        item_id = item.table_id
    else:
        item_type_order = 2
        item_id = item.image_id
    return (
        source_index if source_index is not None else 10**9,
        item_type_order,
        item_id,
    )
