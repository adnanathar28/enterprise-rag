import threading
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from tests.unit.test_document_indexing_service import saved_document, service_with
from tests.unit.test_documents_api import FakeDocumentPersistenceService, FakeIngestionService

from brd_knowledge.api.dependencies import (
    document_indexing_service_dependency,
    document_persistence_service_dependency,
    file_intake_service_dependency,
    ingestion_service_dependency,
)
from brd_knowledge.core.exceptions import (
    DocumentIndexingError,
    DocumentNotFoundError,
    DocumentNotIndexableError,
    ParserError,
)
from brd_knowledge.main import app
from brd_knowledge.schemas.source_file import StoredSourceFile
from brd_knowledge.services.file_intake_service import FileIntakeService


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_index_endpoint_returns_existing_document_summary():
    service, _, provider, _ = service_with(saved_document())
    app.dependency_overrides[document_indexing_service_dependency] = lambda: service
    with TestClient(app) as client:
        response = client.post("/documents/doc-1/index")
    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-1"
    assert response.json()["indexing"]["status"] == "ready"
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (DocumentNotFoundError("Document not found."), 404),
        (DocumentNotIndexableError("No searchable text was extracted."), 422),
        (DocumentIndexingError("Search preparation failed; retry preparation."), 503),
    ],
)
def test_index_endpoint_maps_failures(failure, status):
    service = Mock()
    service.index.side_effect = failure
    app.dependency_overrides[document_indexing_service_dependency] = lambda: service
    with TestClient(app) as client:
        response = client.post("/documents/doc-1/index")
    assert response.status_code == status
    assert response.json()["detail"] == str(failure)


@pytest.mark.parametrize("mode", ["exception", "failed", "partial_success"])
def test_ingestion_parse_outcomes_are_explicit_and_preserve_returned_output(tmp_path, mode):
    persistence = FakeDocumentPersistenceService()
    ingestion = FakeIngestionService()
    original = ingestion.ingest_with_result

    def parse(path, options):
        if mode == "exception":
            raise ParserError("private filesystem path")
        result = original(path, options)
        result.parse_status = mode
        result.parsed_document.parser_metadata.parse_status = mode
        return result

    ingestion.ingest_with_result = parse
    app.dependency_overrides[ingestion_service_dependency] = lambda: ingestion
    app.dependency_overrides[document_persistence_service_dependency] = lambda: persistence
    app.dependency_overrides[file_intake_service_dependency] = lambda: FileIntakeService(
        tmp_path,
        {".pdf"},
        1024,
    )
    with TestClient(app) as client:
        response = client.post("/documents/ingest", files={"file": ("sample.pdf", b"synthetic")})
    assert response.status_code == (200 if mode == "partial_success" else 422)
    assert "private filesystem path" not in response.text
    if mode == "exception":
        assert persistence.calls == []
    else:
        assert len(persistence.calls) == 1
        payload = response.json() if mode == "partial_success" else response.json()["detail"]
        assert payload["parse_status"] == mode
        assert payload["document_id"] == "doc-001"


def test_ingestion_keeps_pdf_only_upload_validation(tmp_path):
    app.dependency_overrides[file_intake_service_dependency] = lambda: FileIntakeService(
        tmp_path,
        {".pdf"},
        1024,
    )
    ingestion = Mock()
    app.dependency_overrides[ingestion_service_dependency] = lambda: ingestion
    app.dependency_overrides[document_persistence_service_dependency] = lambda: (
        FakeDocumentPersistenceService()
    )
    with TestClient(app) as client:
        response = client.post("/documents/ingest", files={"file": ("sample.docx", b"synthetic")})
    assert response.status_code == 400
    ingestion.ingest_with_result.assert_not_called()


def test_blocking_ingestion_work_runs_outside_event_loop(tmp_path):
    thread_ids = {}
    ingestion = FakeIngestionService()
    original = ingestion.ingest_with_result

    async def dependency():
        thread_ids["event_loop"] = threading.get_ident()
        return ingestion

    def parse(path, options):
        thread_ids["parser"] = threading.get_ident()
        return original(path, options)

    ingestion.ingest_with_result = parse
    intake = Mock()
    intake.store.return_value = StoredSourceFile(
        original_filename="sample.pdf",
        stored_filename="sample.pdf",
        stored_path=tmp_path / "sample.pdf",
        size_bytes=1,
        extension=".pdf",
    )
    app.dependency_overrides[ingestion_service_dependency] = dependency
    app.dependency_overrides[file_intake_service_dependency] = lambda: intake
    app.dependency_overrides[document_persistence_service_dependency] = lambda: (
        FakeDocumentPersistenceService()
    )
    with TestClient(app) as client:
        response = client.post("/documents/ingest", files={"file": ("sample.pdf", b"synthetic")})
    assert response.status_code == 200
    assert thread_ids["parser"] != thread_ids["event_loop"]
