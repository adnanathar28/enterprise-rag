from pathlib import Path

from brd_knowledge.parsing.base import DocumentParser
from brd_knowledge.schemas.document import ParsedDocument


class IngestionService:
    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._parsers = parsers

    def ingest(self, file_path: Path) -> ParsedDocument:
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser.parse(file_path)
        msg = f"No parser registered for file: {file_path}"
        raise ValueError(msg)
