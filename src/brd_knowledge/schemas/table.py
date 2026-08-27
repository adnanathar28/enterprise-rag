from pydantic import BaseModel, Field

from brd_knowledge.schemas.source import BoundingBox, SourceReference


class TableCell(BaseModel):
    cell_id: str | None = None
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    text: str
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    bounding_box: BoundingBox | None = None
    source: SourceReference | None = None
    is_header: bool = False


class TableRow(BaseModel):
    row_index: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)


class ParsedTable(BaseModel):
    table_id: str
    page_number: int | None = Field(default=None, ge=1)
    page_numbers: list[int] = Field(default_factory=list)
    reading_order_index: int | None = Field(default=None, ge=0)
    rows: list[TableRow] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    caption: str | None = None
    bounding_box: BoundingBox | None = None
    source: SourceReference | None = None
    native_text: str | None = None
    native_text_coverage: float | None = Field(default=None, ge=0, le=1)
    quality_notes: list[str] = Field(default_factory=list)
