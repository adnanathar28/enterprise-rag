from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from brd_knowledge.api.dependencies import (
    document_persistence_service_dependency,
    file_intake_service_dependency,
    ingestion_service_dependency,
)
from brd_knowledge.parsing.options import ParseOptions
from brd_knowledge.schemas.ingestion import IngestionSummary
from brd_knowledge.schemas.persisted_document import (
    PersistedDocumentSummary,
    PersistedParsedDocument,
)
from brd_knowledge.services.document_persistence_service import DocumentPersistenceService
from brd_knowledge.services.file_intake_service import FileIntakeService
from brd_knowledge.services.ingestion_service import IngestionService

router = APIRouter()
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


@router.get("/", response_model=list[PersistedDocumentSummary])
def list_documents(
    document_persistence_service: Annotated[
        DocumentPersistenceService,
        Depends(document_persistence_service_dependency),
    ],
) -> list[PersistedDocumentSummary]:
    return [
        PersistedDocumentSummary.from_model(document)
        for document in document_persistence_service.list_documents()
    ]


@router.get("/{document_id}", response_model=PersistedDocumentSummary)
def get_document(
    document_id: str,
    document_persistence_service: Annotated[
        DocumentPersistenceService,
        Depends(document_persistence_service_dependency),
    ],
) -> PersistedDocumentSummary:
    document = document_persistence_service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    return PersistedDocumentSummary.from_model(document)


@router.get("/{document_id}/parsed", response_model=PersistedParsedDocument)
def get_parsed_document(
    document_id: str,
    document_persistence_service: Annotated[
        DocumentPersistenceService,
        Depends(document_persistence_service_dependency),
    ],
) -> PersistedParsedDocument:
    parsed_document = document_persistence_service.get_parsed_document_json(document_id)
    if parsed_document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    return PersistedParsedDocument(document_id=document_id, parsed_document=parsed_document)


@router.post("/ingest", response_model=IngestionSummary)
async def ingest_document(
    file: Annotated[UploadFile, File()],
    file_intake_service: Annotated[FileIntakeService, Depends(file_intake_service_dependency)],
    ingestion_service: Annotated[IngestionService, Depends(ingestion_service_dependency)],
    document_persistence_service: Annotated[
        DocumentPersistenceService,
        Depends(document_persistence_service_dependency),
    ],
    split_pages: Annotated[bool, Form()] = False,
    page_start: Annotated[int | None, Form()] = None,
    page_end: Annotated[int | None, Form()] = None,
    page_timeout_seconds: Annotated[float | None, Form()] = None,
) -> IngestionSummary:
    try:
        parse_options = _build_parse_options(
            split_pages=split_pages,
            page_start=page_start,
            page_end=page_end,
            page_timeout_seconds=page_timeout_seconds,
        )
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / "upload"
            await _write_upload_to_path(file, temporary_path)
            stored_file = file_intake_service.store(
                temporary_path,
                original_filename=file.filename,
            )
        result = ingestion_service.ingest_with_result(stored_file.stored_path, parse_options)
        document_persistence_service.save_ingestion_result(stored_file, result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = IngestionSummary.from_ingestion_result(result)
    return summary.model_copy(update={"filename": stored_file.original_filename})


def _build_parse_options(
    split_pages: bool,
    page_start: int | None,
    page_end: int | None,
    page_timeout_seconds: float | None,
) -> ParseOptions:
    if page_start is None and page_end is None:
        page_range = None
    elif page_start is not None and page_end is not None:
        page_range = (page_start, page_end)
    else:
        raise ValueError("Both page_start and page_end are required when specifying a page range.")

    return ParseOptions(
        page_range=page_range,
        split_pages=split_pages,
        page_timeout_seconds=page_timeout_seconds,
    )


async def _write_upload_to_path(upload_file: UploadFile, destination: Path) -> None:
    with destination.open("wb") as output_file:
        while chunk := await upload_file.read(UPLOAD_CHUNK_SIZE_BYTES):
            output_file.write(chunk)
