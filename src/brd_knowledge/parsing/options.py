from pydantic import BaseModel, Field, model_validator


class ParseOptions(BaseModel):
    page_range: tuple[int, int] | None = None
    split_pages: bool = False
    page_timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_page_options(self) -> "ParseOptions":
        if self.page_range is not None:
            start_page, end_page = self.page_range
            if start_page < 1 or end_page < 1:
                raise ValueError("Page range values must be positive 1-based page numbers.")
            if start_page > end_page:
                raise ValueError(
                    f"Invalid page range: start page {start_page} is after end page {end_page}."
                )
        if self.split_pages and self.page_range is None:
            raise ValueError("Split-page parsing requires a page range.")
        if self.page_range is not None and not self.split_pages:
            raise ValueError("A page range requires split_pages=true.")
        return self
