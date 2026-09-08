from pathlib import Path
from typing import Any

import pytest

from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.schemas.document import DocumentMetadata, Page, ParsedDocument, ParserMetadata
from brd_knowledge.schemas.ingestion import IngestionResult
from brd_knowledge.schemas.source_file import StoredSourceFile
from brd_knowledge.services.document_persistence_service import DocumentPersistenceService


class FakeSession:
    def __init__(self, scalar_results: list[int] | None = None) -> None:
        self.added: list[Any] = []
        self.committed = False
        self.refreshed: list[Any] = []
        self.scalar_results = list(scalar_results or [])
        self.scalar_calls: list[Any] = []

    def add(self, value: Any) -> None:
        self.added.append(value)

    def commit(self) -> None:
        self.committed = True

    def refresh(self, value: Any) -> None:
        self.refreshed.append(value)

    def scalar(self, statement: Any) -> int:
        self.scalar_calls.append(statement)
        return self.scalar_results.pop(0)


def test_save_ingestion_result_persists_document_metadata_and_json() -> None:
    session = FakeSession()
    service = DocumentPersistenceService(session)  # type: ignore[arg-type]
    stored_file = StoredSourceFile(
        original_filename="sample.pdf",
        stored_filename="uuid.pdf",
        stored_path=Path("data/source_files/uuid.pdf"),
        size_bytes=123,
        extension=".pdf",
    )
    parsed_document = ParsedDocument(
        metadata=DocumentMetadata(
            document_id="doc-001",
            filename="uuid.pdf",
            file_type="pdf",
            page_count=1,
            parser_name="docling",
            parser_version="2.114.0",
        ),
        parser_metadata=ParserMetadata(
            parser_name="docling",
            parser_version="2.114.0",
            parse_status="success",
        ),
        pages=[Page(page_number=1, parse_status="success")],
    )
    ingestion_result = IngestionResult.from_parsed_document(parsed_document)

    document = service.save_ingestion_result(stored_file, ingestion_result)

    assert session.added == [document]
    assert session.committed is True
    assert session.refreshed == [document]
    assert document.document_id == "doc-001"
    assert document.filename == "sample.pdf"
    assert document.original_filename == "sample.pdf"
    assert document.stored_filename == "uuid.pdf"
    assert document.stored_path == str(Path("data/source_files/uuid.pdf"))
    assert document.file_type == "pdf"
    assert document.size_bytes == 123
    assert document.page_count == 1
    assert document.parse_status == "success"
    assert document.parser_name == "docling"
    assert document.parser_version == "2.114.0"
    assert document.parsed_document_json["metadata"]["document_id"] == "doc-001"


@pytest.mark.parametrize(
    ("counts", "expected_status"),
    [
        ([0, 0], "not_indexed"),
        ([4, 4], "ready"),
        ([4, 2], "needs_reindex"),
        ([4, 0], "needs_reindex"),
    ],
)
def test_get_indexing_summary_reports_query_readiness(
    counts: list[int],
    expected_status: str,
) -> None:
    session = FakeSession(counts)
    service = DocumentPersistenceService(session)  # type: ignore[arg-type]
    configuration = GteModernBertEmbeddingProvider().configuration

    result = service.get_indexing_summary("doc-001", configuration)

    assert result.status == expected_status
    assert result.total_chunk_count == counts[0]
    assert result.compatible_chunk_count == counts[1]
    assert len(session.scalar_calls) == 2
