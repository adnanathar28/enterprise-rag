from pathlib import Path
from shutil import copy2

from fastapi.testclient import TestClient

from brd_knowledge.api.dependencies import (
    document_persistence_service_dependency,
    file_intake_service_dependency,
    ingestion_service_dependency,
)
from brd_knowledge.main import app
from brd_knowledge.parsing.options import ParseOptions
from brd_knowledge.schemas.document import DocumentMetadata, Page, ParsedDocument, ParserMetadata
from brd_knowledge.schemas.ingestion import IngestionResult
from brd_knowledge.schemas.source_file import StoredSourceFile


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
    def __init__(self) -> None:
        self.calls: list[tuple[StoredSourceFile, IngestionResult]] = []

    def save_ingestion_result(
        self,
        stored_file: StoredSourceFile,
        ingestion_result: IngestionResult,
    ) -> None:
        self.calls.append((stored_file, ingestion_result))


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
