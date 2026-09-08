from pathlib import Path
from shutil import copy2
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from tests.unit.test_grounded_answer_generation import FakeProvider

from brd_knowledge.api.dependencies import (
    document_persistence_service_dependency,
    embedding_provider_dependency,
    file_intake_service_dependency,
    ingestion_service_dependency,
    question_answering_service_dependency,
    settings_dependency,
)
from brd_knowledge.core.config import Settings
from brd_knowledge.main import app
from brd_knowledge.parsing.options import ParseOptions
from brd_knowledge.schemas.document import DocumentMetadata, Page, ParsedDocument, ParserMetadata
from brd_knowledge.schemas.ingestion import IngestionResult
from brd_knowledge.schemas.persisted_document import DocumentIndexingSummary
from brd_knowledge.schemas.retrieval import RetrievedChunk
from brd_knowledge.schemas.source_file import StoredSourceFile
from brd_knowledge.services.question_answering_service import QuestionAnsweringService


class FakeFileIntakeService:
    def __init__(self, storage_dir: Path, fail_with: Exception | None = None) -> None:
        self.storage_dir = storage_dir
        self.fail_with = fail_with
        self.calls: list[tuple[Path, str | None]] = []

    def store(self, source_path: Path, original_filename: str | None = None) -> StoredSourceFile:
        self.calls.append((source_path, original_filename))
        if self.fail_with is not None:
            raise self.fail_with
        stored_path = self.storage_dir / "stored.pdf"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        copy2(source_path, stored_path)
        return StoredSourceFile(
            original_filename=original_filename or source_path.name,
            stored_filename=stored_path.name,
            stored_path=stored_path,
            size_bytes=stored_path.stat().st_size,
            extension=".pdf",
        )


class FakeIngestionService:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, ParseOptions | None]] = []

    def ingest_with_result(
        self,
        file_path: Path,
        options: ParseOptions | None = None,
    ) -> IngestionResult:
        self.calls.append((file_path, options))
        parsed_document = ParsedDocument(
            metadata=DocumentMetadata(
                document_id="doc-001",
                filename=file_path.name,
                file_type="pdf",
                page_count=2,
                parser_name="fake",
            ),
            parser_metadata=ParserMetadata(
                parser_name="fake",
                parse_status="success",
            ),
            pages=[
                Page(page_number=1, parse_status="success"),
                Page(page_number=2, parse_status="success"),
            ],
        )
        return IngestionResult.from_parsed_document(parsed_document)


class FakeDocumentPersistenceService:
    def __init__(
        self,
        indexing_status: Literal["not_indexed", "ready", "needs_reindex"] = "ready",
    ) -> None:
        self.calls: list[tuple[StoredSourceFile, IngestionResult]] = []
        self.indexing_status = indexing_status
        self.documents = [
            SimpleNamespace(
                document_id="doc-001",
                filename="sample.pdf",
                original_filename="sample.pdf",
                stored_filename="stored.pdf",
                file_type="pdf",
                size_bytes=123,
                page_count=2,
                parse_status="success",
                parser_name="fake",
                parser_version="1.0",
                created_at="2026-07-27T00:00:00",
            )
        ]
        self.parsed_documents = {
            "doc-001": {
                "metadata": {
                    "document_id": "doc-001",
                    "filename": "stored.pdf",
                    "file_type": "pdf",
                    "page_count": 2,
                },
                "pages": [{"page_number": 1}, {"page_number": 2}],
            }
        }

    def save_ingestion_result(
        self,
        stored_file: StoredSourceFile,
        ingestion_result: IngestionResult,
    ) -> None:
        self.calls.append((stored_file, ingestion_result))

    def list_documents(self) -> list[SimpleNamespace]:
        return self.documents

    def get_document(self, document_id: str) -> SimpleNamespace | None:
        return next(
            (document for document in self.documents if document.document_id == document_id),
            None,
        )

    def get_parsed_document_json(self, document_id: str) -> dict[str, Any] | None:
        return self.parsed_documents.get(document_id)

    def get_indexing_summary(
        self,
        document_id: str,
        configuration: object,
    ) -> DocumentIndexingSummary:
        del document_id, configuration
        count = 1 if self.indexing_status == "ready" else 0
        return DocumentIndexingSummary(
            status=self.indexing_status,
            compatible_chunk_count=count,
            total_chunk_count=count,
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, document_id))
        return [
            RetrievedChunk(
                rank=1,
                cosine_distance=0.1,
                similarity=0.9,
                chunk_id="chunk-1",
                document_id=document_id or "doc-001",
                content_type="prose",
                text="Audit records must be retained.",
                page_start=7,
                page_end=7,
            )
        ]


def test_ingest_document_upload_returns_summary(tmp_path: Path) -> None:
    intake_service = FakeFileIntakeService(tmp_path / "source_files")
    ingestion_service = FakeIngestionService()
    persistence_service = FakeDocumentPersistenceService()
    app.dependency_overrides[file_intake_service_dependency] = lambda: intake_service
    app.dependency_overrides[ingestion_service_dependency] = lambda: ingestion_service
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service

    try:
        client = TestClient(app)
        response = client.post(
            "/documents/ingest",
            files={"file": ("sample.pdf", b"%PDF-1.7 synthetic", "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-001",
        "filename": "sample.pdf",
        "parse_status": "success",
        "page_count": 2,
        "block_count": 0,
        "table_count": 0,
        "table_cell_count": 0,
        "image_count": 0,
        "section_count": 0,
        "diagnostic_count": 0,
        "failed_page_count": 0,
    }
    assert intake_service.calls[0][1] == "sample.pdf"
    assert ingestion_service.calls[0][0] == tmp_path / "source_files" / "stored.pdf"
    assert persistence_service.calls[0][0].stored_filename == "stored.pdf"
    assert persistence_service.calls[0][1].document_id == "doc-001"


def test_ingest_document_passes_split_page_options(tmp_path: Path) -> None:
    intake_service = FakeFileIntakeService(tmp_path / "source_files")
    ingestion_service = FakeIngestionService()
    persistence_service = FakeDocumentPersistenceService()
    app.dependency_overrides[file_intake_service_dependency] = lambda: intake_service
    app.dependency_overrides[ingestion_service_dependency] = lambda: ingestion_service
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service

    try:
        client = TestClient(app)
        response = client.post(
            "/documents/ingest",
            data={
                "split_pages": "true",
                "page_start": "1",
                "page_end": "3",
                "page_timeout_seconds": "10",
            },
            files={"file": ("sample.pdf", b"%PDF-1.7 synthetic", "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    options = ingestion_service.calls[0][1]
    assert options is not None
    assert options.split_pages is True
    assert options.page_range == (1, 3)
    assert options.page_timeout_seconds == 10


def test_ingest_document_rejects_invalid_page_range(tmp_path: Path) -> None:
    intake_service = FakeFileIntakeService(tmp_path / "source_files")
    ingestion_service = FakeIngestionService()
    persistence_service = FakeDocumentPersistenceService()
    app.dependency_overrides[file_intake_service_dependency] = lambda: intake_service
    app.dependency_overrides[ingestion_service_dependency] = lambda: ingestion_service
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service

    try:
        client = TestClient(app)
        response = client.post(
            "/documents/ingest",
            data={"split_pages": "true", "page_start": "1"},
            files={"file": ("sample.pdf", b"%PDF-1.7 synthetic", "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "Both page_start and page_end are required" in response.json()["detail"]
    assert intake_service.calls == []
    assert ingestion_service.calls == []
    assert persistence_service.calls == []


def test_ingest_document_returns_intake_validation_errors(tmp_path: Path) -> None:
    intake_service = FakeFileIntakeService(
        tmp_path / "source_files",
        fail_with=ValueError("Unsupported file extension '.exe'. Expected one of: .pdf"),
    )
    ingestion_service = FakeIngestionService()
    persistence_service = FakeDocumentPersistenceService()
    app.dependency_overrides[file_intake_service_dependency] = lambda: intake_service
    app.dependency_overrides[ingestion_service_dependency] = lambda: ingestion_service
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service

    try:
        client = TestClient(app)
        response = client.post(
            "/documents/ingest",
            files={"file": ("sample.exe", b"bad", "application/octet-stream")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]
    assert ingestion_service.calls == []
    assert persistence_service.calls == []


def test_list_documents_returns_persisted_summaries() -> None:
    persistence_service = FakeDocumentPersistenceService()
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service
    app.dependency_overrides[embedding_provider_dependency] = lambda: SimpleNamespace(
        configuration=object()
    )

    try:
        client = TestClient(app)
        response = client.get("/documents/")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()[0]["document_id"] == "doc-001"
    assert response.json()[0]["filename"] == "sample.pdf"
    assert response.json()[0]["parse_status"] == "success"
    assert response.json()[0]["indexing"]["status"] == "ready"


def test_get_document_returns_persisted_summary() -> None:
    persistence_service = FakeDocumentPersistenceService()
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service
    app.dependency_overrides[embedding_provider_dependency] = lambda: SimpleNamespace(
        configuration=object()
    )

    try:
        client = TestClient(app)
        response = client.get("/documents/doc-001")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-001"
    assert response.json()["stored_filename"] == "stored.pdf"


def test_get_document_returns_404_for_missing_document() -> None:
    persistence_service = FakeDocumentPersistenceService()
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service
    app.dependency_overrides[embedding_provider_dependency] = lambda: SimpleNamespace(
        configuration=object()
    )

    try:
        client = TestClient(app)
        response = client.get("/documents/missing")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found: missing"


def test_get_parsed_document_returns_stored_json() -> None:
    persistence_service = FakeDocumentPersistenceService()
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service

    try:
        client = TestClient(app)
        response = client.get("/documents/doc-001/parsed")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-001"
    assert response.json()["parsed_document"]["metadata"]["document_id"] == "doc-001"


def test_ask_document_question_returns_grounded_answer_and_evidence() -> None:
    persistence_service = FakeDocumentPersistenceService()
    retriever = FakeRetriever()
    provider = FakeProvider()
    question_service = QuestionAnsweringService(
        retriever,
        Settings(_env_file=None),
        provider_factory=lambda *_args, **_kwargs: provider,
    )
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service
    app.dependency_overrides[embedding_provider_dependency] = lambda: SimpleNamespace(
        configuration=object()
    )
    app.dependency_overrides[question_answering_service_dependency] = lambda: question_service

    try:
        client = TestClient(app)
        response = client.post(
            "/documents/doc-001/questions",
            json={"question": "What must be retained?", "provider": "local_qwen"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]["citations"][0]["chunk_id"] == "chunk-1"
    assert payload["answer"]["citations"][0]["page_start"] == 7
    assert payload["context"]["evidence"][0]["evidence_id"] == "E1"
    assert payload["retrieved_chunks"][0]["similarity"] == 0.9
    assert retriever.calls == [("What must be retained?", 5, "doc-001")]


def test_ask_document_question_rejects_document_that_is_not_indexed() -> None:
    persistence_service = FakeDocumentPersistenceService(indexing_status="not_indexed")
    question_service = MagicMock()
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence_service
    app.dependency_overrides[embedding_provider_dependency] = lambda: SimpleNamespace(
        configuration=object()
    )
    app.dependency_overrides[question_answering_service_dependency] = lambda: question_service

    try:
        client = TestClient(app)
        response = client.post(
            "/documents/doc-001/questions",
            json={"question": "What must be retained?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert response.json()["detail"].endswith("not_indexed")
    question_service.answer.assert_not_called()


def test_capabilities_reports_provider_configuration_without_secrets() -> None:
    settings = Settings(
        _env_file=None,
        LLM_PROVIDER="local_qwen",
        GEMINI_API_KEY="secret-test-key",
    )
    app.dependency_overrides[settings_dependency] = lambda: settings

    try:
        client = TestClient(app)
        response = client.get("/capabilities")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["providers"][0]["provider"] == "local_qwen"
    assert payload["providers"][0]["is_default"] is True
    assert payload["providers"][1]["provider"] == "gemini"
    assert payload["providers"][1]["configured"] is True
    assert "secret-test-key" not in response.text
