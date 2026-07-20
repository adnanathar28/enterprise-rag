from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    page_number: int = Field(ge=1)
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)


class TableCell(BaseModel):
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    text: str
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    bounding_box: BoundingBox | None = None


class ParsedTable(BaseModel):
    table_id: str
    page_number: int = Field(ge=1)
    cells: list[TableCell] = Field(default_factory=list)
    caption: str | None = None
    bounding_box: BoundingBox | None = None
