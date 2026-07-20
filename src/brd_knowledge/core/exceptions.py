class BRDKnowledgeError(Exception):
    """Base exception for application-specific errors."""


class ParserError(BRDKnowledgeError):
    """Raised when a document parser cannot process a file."""
