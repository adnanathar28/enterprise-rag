from copy import deepcopy
from datetime import datetime
from unittest.mock import Mock

import pytest
from tests.unit.test_chunk_embedding_service import FakeProvider, FakeSession

from brd_knowledge.core.exceptions import (
    DocumentIndexingError,
    DocumentNotFoundError,
    DocumentNotIndexableError,
)
from brd_knowledge.database.models.document import Document
from brd_knowledge.schemas.document import (
    DocumentMetadata,
    ParsedDocument,
    ParserMetadata,
    TextBlock,
)
from brd_knowledge.schemas.persisted_document import DocumentIndexingSummary
from brd_knowledge.services.chunk_embedding_service import ChunkEmbeddingService
from brd_knowledge.services.document_indexing_service import DocumentIndexingService


def saved_document() -> Document:
    parsed = ParsedDocument(
        metadata=DocumentMetadata(
            document_id="doc-1",
            filename="sample.pdf",
            file_type="pdf",
            page_count=1,
        ),
        parser_metadata=ParserMetadata(parser_name="fake", parse_status="success"),
        blocks=[TextBlock(block_id="block-1", page_number=1, text="Keep records for five years.")],
    )
    return Document(
        document_id="doc-1",
        filename="sample.pdf",
        original_filename="sample.pdf",
        stored_filename="sample.pdf",
        stored_path="/tmp/sample.pdf",
        file_type="pdf",
        size_bytes=12,
        page_count=1,
        parse_status="success",
        created_at=datetime(2026, 9, 8),
        parsed_document_json=parsed.model_dump(mode="json"),
    )


def service_with(document, provider=None):
    session = FakeSession()
    provider = provider or FakeProvider()
    persistence = Mock()
    persistence.get_document.return_value = document
    persistence.get_indexing_summary.side_effect = lambda *_: DocumentIndexingSummary(
        status="ready" if session.rows else "not_indexed",
        total_chunk_count=len(session.rows),
        compatible_chunk_count=len(session.rows),
    )
    service = DocumentIndexingService(
        persistence,
        ChunkEmbeddingService(session, provider),
        provider.configuration,
    )
    return service, session, provider, persistence


def test_index_saved_parse_and_repeat_without_reembedding():
    document = saved_document()
    service, session, provider, _ = service_with(document)
    first = service.index("doc-1")
    second = service.index("doc-1")
    assert first.indexing.status == second.indexing.status == "ready"
    assert len(session.rows) == 1
    assert provider.calls == [["Keep records for five years."]]
    assert session.rows[0].source_block_ids == ["block-1"]
    assert session.rows[0].page_start == 1
    assert session.rows[0].document_id == "doc-1"


def test_failure_preserves_saved_parse_and_retry_succeeds():
    document = saved_document()
    original = deepcopy(document.parsed_document_json)
    provider = FakeProvider()
    provider.embed_documents = Mock(side_effect=RuntimeError("private SDK details"))
    service, session, _, persistence = service_with(document, provider)
    with pytest.raises(DocumentIndexingError, match="retry preparation"):
        service.index("doc-1")
    assert session.rollbacks == 1
    assert session.commits == 0
    assert session.rows == []
    assert document.parsed_document_json == original
    assert persistence.get_document("doc-1") is document
    provider.embed_documents = FakeProvider().embed_documents
    assert service.index("doc-1").indexing.status == "ready"
    assert document.parsed_document_json == original


@pytest.mark.parametrize("invalid", ["missing", "malformed", "failed", "empty", "mismatched"])
def test_rejects_unusable_input_before_embedding(invalid):
    document = saved_document()
    if invalid == "missing":
        document = None
    elif invalid == "malformed":
        document.parsed_document_json = {}
    elif invalid == "failed":
        document.parse_status = "failed"
        document.parsed_document_json["parser_metadata"]["parse_status"] = "failed"
    elif invalid == "empty":
        document.parsed_document_json["blocks"] = []
    else:
        document.parsed_document_json["metadata"]["document_id"] = "other-document"
    service, session, provider, _ = service_with(document)
    expected = DocumentNotFoundError if invalid == "missing" else DocumentNotIndexableError
    with pytest.raises(expected):
        service.index("doc-1")
    assert provider.calls == []
    assert session.commits == 0


def test_partial_parse_remains_visible_after_indexing():
    document = saved_document()
    document.parse_status = "partial_success"
    document.parsed_document_json["parser_metadata"]["parse_status"] = "partial_success"
    service, _, _, _ = service_with(document)
    result = service.index("doc-1")
    assert result.indexing.status == "ready"
    assert result.parse_status == "partial_success"
