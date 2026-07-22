from typing import Literal

from pydantic import BaseModel, Field

CoordinateOrigin = Literal["top_left", "bottom_left", "unknown"]


class BoundingBox(BaseModel):
    page_number: int = Field(ge=1)
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    coordinate_origin: CoordinateOrigin = "unknown"


class SourceReference(BaseModel):
    document_id: str
    page_number: int | None = Field(default=None, ge=1)
    parser_item_id: str | None = None
    section_id: str | None = None
    block_id: str | None = None
    table_id: str | None = None
    cell_id: str | None = None
    reading_order_index: int | None = Field(default=None, ge=0)
    text_excerpt: str | None = None
    bounding_box: BoundingBox | None = None
