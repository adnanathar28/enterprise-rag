from abc import ABC, abstractmethod
from pathlib import Path

from brd_knowledge.schemas.document import ParsedDocument


class DocumentParser(ABC):
    parser_name: str

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Return whether this parser supports the provided file."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a source document into the normalized document schema."""
