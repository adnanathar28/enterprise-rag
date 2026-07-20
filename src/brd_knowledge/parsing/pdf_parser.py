from pathlib import Path

from brd_knowledge.core.exceptions import ParserError
from brd_knowledge.parsing.base import DocumentParser
from brd_knowledge.schemas.document import ParsedDocument


class DiagnosticPdfParser(DocumentParser):
    parser_name = "diagnostic-pdf"

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> ParsedDocument:
        raise ParserError("Fallback PDF parsing is not implemented in the scaffolding phase.")
