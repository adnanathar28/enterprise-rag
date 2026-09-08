from pydantic import ValidationError

from brd_knowledge.chunking import StructureAwareChunker
from brd_knowledge.core.exceptions import (
    DocumentIndexingError,
    DocumentNotFoundError,
    DocumentNotIndexableError,
)
from brd_knowledge.embeddings.base import EmbeddingConfiguration
from brd_knowledge.schemas.document import ParsedDocument
from brd_knowledge.schemas.persisted_document import PersistedDocumentSummary
from brd_knowledge.services.chunk_embedding_service import ChunkEmbeddingService
from brd_knowledge.services.document_persistence_service import DocumentPersistenceService


class DocumentIndexingService:
    """Prepare saved parsing output for search without reparsing or replacing the document."""

    def __init__(
        self,
        persistence: DocumentPersistenceService,
        embeddings: ChunkEmbeddingService,
        configuration: EmbeddingConfiguration,
    ) -> None:
        self._persistence = persistence
        self._embeddings = embeddings
        self._configuration = configuration

    def index(self, document_id: str) -> PersistedDocumentSummary:
        document = self._persistence.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError("Document not found.")
        try:
            parsed = ParsedDocument.model_validate(document.parsed_document_json)
        except ValidationError as exc:
            raise DocumentNotIndexableError("Saved parsing output is invalid.") from exc
        status = parsed.parser_metadata.parse_status if parsed.parser_metadata else None
        if document.parse_status not in {"success", "partial_success"} or status not in {
            "success",
            "partial_success",
        }:
            raise DocumentNotIndexableError("Document parsing did not complete successfully.")
        if parsed.metadata.document_id != document_id:
            raise DocumentNotIndexableError("Saved parsing output belongs to another document.")
        try:
            chunks = StructureAwareChunker().chunk(parsed)
            if not chunks or not any(chunk.text.strip() for chunk in chunks):
                raise DocumentNotIndexableError("No searchable text was extracted.")
            # Synchronization owns its transaction and rolls back on failure.
            # The parsed document was committed by ingestion in an earlier request.
            self._embeddings.synchronize(document_id, chunks)
        except DocumentNotIndexableError:
            raise
        except Exception as exc:
            raise DocumentIndexingError(
                "Search preparation failed. The parsed document is saved; retry preparation."
            ) from exc
        return PersistedDocumentSummary.from_model(
            document,
            self._persistence.get_indexing_summary(document_id, self._configuration),
        )
