from pydantic import BaseModel

from brd_knowledge.schemas.document import ParsedDocument, ParseStatus


class IngestionResult(BaseModel):
    document_id: str
    filename: str
    parse_status: ParseStatus | None = None
    page_count: int
    block_count: int
    table_count: int
    table_cell_count: int
    image_count: int
    section_count: int
    diagnostic_count: int
    failed_page_count: int
    parsed_document: ParsedDocument

    @classmethod
    def from_parsed_document(cls, parsed_document: ParsedDocument) -> "IngestionResult":
        parse_status = (
            parsed_document.parser_metadata.parse_status
            if parsed_document.parser_metadata is not None
            else None
        )
        return cls(
            document_id=parsed_document.metadata.document_id,
            filename=parsed_document.metadata.filename,
            parse_status=parse_status,
            page_count=len(parsed_document.pages),
            block_count=len(parsed_document.blocks),
            table_count=len(parsed_document.tables),
            table_cell_count=sum(len(table.cells) for table in parsed_document.tables),
            image_count=len(parsed_document.images),
            section_count=len(parsed_document.sections),
            diagnostic_count=len(parsed_document.diagnostics),
            failed_page_count=sum(
                1 for page in parsed_document.pages if page.parse_status == "failed"
            ),
            parsed_document=parsed_document,
        )


class IngestionSummary(BaseModel):
    document_id: str
    filename: str
    parse_status: ParseStatus | None = None
    page_count: int
    block_count: int
    table_count: int
    table_cell_count: int
    image_count: int
    section_count: int
    diagnostic_count: int
    failed_page_count: int

    @classmethod
    def from_ingestion_result(cls, result: IngestionResult) -> "IngestionSummary":
        return cls(
            document_id=result.document_id,
            filename=result.filename,
            parse_status=result.parse_status,
            page_count=result.page_count,
            block_count=result.block_count,
            table_count=result.table_count,
            table_cell_count=result.table_cell_count,
            image_count=result.image_count,
            section_count=result.section_count,
            diagnostic_count=result.diagnostic_count,
            failed_page_count=result.failed_page_count,
        )
