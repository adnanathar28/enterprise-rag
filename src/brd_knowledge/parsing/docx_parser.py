from pathlib import Path

from brd_knowledge.core.exceptions import ParserError
from brd_knowledge.parsing.base import DocumentParser
from brd_knowledge.schemas.document import ParsedDocument


class DocxDocumentParser(DocumentParser):
    parser_name = "python-docx"

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".docx"

    def parse(self, file_path: Path) -> ParsedDocument:
        raise ParserError("DOCX parsing is reserved for a later phase.")
