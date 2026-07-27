from pathlib import Path
from typing import Protocol

from brd_knowledge.parsing.base import DocumentParser
from brd_knowledge.parsing.options import ParseOptions
from brd_knowledge.parsing.split_pages import process_split_pages
from brd_knowledge.schemas.document import ParsedDocument


class SplitPageProcessor(Protocol):
    def __call__(
        self,
        document_path: Path,
        page_range: tuple[int, int],
        page_timeout_seconds: float | None = None,
    ) -> ParsedDocument:
        """Parse a document one page at a time and merge the results."""


class IngestionService:
    def __init__(
        self,
        parsers: list[DocumentParser],
        split_page_processor: SplitPageProcessor = process_split_pages,
    ) -> None:
        self._parsers = parsers
        self._split_page_processor = split_page_processor

    def ingest(self, file_path: Path, options: ParseOptions | None = None) -> ParsedDocument:
        parse_options = options or ParseOptions()
        parser = self._select_parser(file_path)
        if parse_options.split_pages:
            if parse_options.page_range is None:
                raise ValueError("Split-page parsing requires a page range.")
            return self._split_page_processor(
                file_path,
                parse_options.page_range,
                parse_options.page_timeout_seconds,
            )
        return parser.parse(file_path)

    def _select_parser(self, file_path: Path) -> DocumentParser:
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser
        msg = f"No parser registered for file: {file_path}"
        raise ValueError(msg)
