from brd_knowledge.parsing.reading_order import stabilize_reading_order
from brd_knowledge.schemas.document import DocumentBlock, ParsedImage
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.schemas.table import ParsedTable


def test_stabilizes_page_local_order_and_preserves_source_indexes() -> None:
    paragraph = DocumentBlock(
        block_id="paragraph",
        page_number=1,
        text="Paragraph",
        reading_order_index=4,
        source=SourceReference(document_id="doc-1", page_number=1, reading_order_index=4),
    )
    table = ParsedTable(
        table_id="table",
        page_number=1,
        reading_order_index=4,
        source=SourceReference(document_id="doc-1", page_number=1, reading_order_index=4),
    )
    image = ParsedImage(
        image_id="image",
        page_number=1,
        reading_order_index=None,
        source=SourceReference(document_id="doc-1", page_number=1, reading_order_index=None),
    )
    next_page = DocumentBlock(
        block_id="next-page",
        page_number=2,
        text="Next page",
        reading_order_index=20,
        source=SourceReference(document_id="doc-1", page_number=2, reading_order_index=20),
    )

    stabilize_reading_order([next_page, paragraph], [table], [image])

    assert paragraph.reading_order_index == 0
    assert table.reading_order_index == 1
    assert image.reading_order_index == 2
    assert next_page.reading_order_index == 0
    assert paragraph.source is not None
    assert paragraph.source.reading_order_index == 4
    assert table.source is not None
    assert table.source.reading_order_index == 4
