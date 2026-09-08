class BRDKnowledgeError(Exception):
    """Base exception for application-specific errors."""


class ParserError(BRDKnowledgeError):
    """Raised when a document parser cannot process a file."""


class DocumentNotFoundError(BRDKnowledgeError):
    """Raised when indexing targets a document that is not persisted."""


class DocumentNotIndexableError(BRDKnowledgeError):
    """Raised when saved parsing output cannot produce searchable evidence."""


class DocumentIndexingError(BRDKnowledgeError):
    """Raised when search preparation fails; the saved document can be retried."""


class GenerationError(BRDKnowledgeError):
    """Base exception for grounded answer generation failures."""


class MalformedGenerationResponse(GenerationError):
    """Raised when the provider output does not match the required schema."""


class InvalidCitationError(GenerationError):
    """Raised when generated citations are absent, unknown, or inconsistent."""


class GenerationProviderError(GenerationError):
    """Raised when the configured model provider fails."""


class PromptBudgetExceeded(GenerationError):
    """Raised before generation when the complete request exceeds the model window."""


class GenerationBlockedError(GenerationProviderError):
    """Raised when the provider blocks or refuses generation."""


class IncompleteGenerationError(GenerationProviderError):
    """Raised when generation stops before a complete answer is available."""
