from pathlib import Path

import pytest

from brd_knowledge.services.file_intake_service import FileIntakeService


def build_service(tmp_path: Path, max_upload_size_bytes: int = 1024) -> FileIntakeService:
    return FileIntakeService(
        storage_dir=tmp_path / "source_files",
        allowed_extensions={".pdf"},
        max_upload_size_bytes=max_upload_size_bytes,
    )


def test_store_accepts_valid_pdf_and_uses_uuid_storage_name(tmp_path: Path) -> None:
    source_path = tmp_path / "upload.tmp"
    source_path.write_bytes(b"%PDF-1.7 synthetic")
    service = build_service(tmp_path)

    stored_file = service.store(source_path, original_filename="Example BRD.PDF")

    assert stored_file.original_filename == "Example BRD.PDF"
    assert stored_file.extension == ".pdf"
    assert stored_file.size_bytes == source_path.stat().st_size
    assert stored_file.stored_filename.endswith(".pdf")
    assert stored_file.stored_filename != "Example BRD.PDF"
    assert len(stored_file.stored_filename) == 36
    assert stored_file.stored_path.exists()
    assert stored_file.stored_path.read_bytes() == source_path.read_bytes()


def test_store_rejects_unsupported_extension(tmp_path: Path) -> None:
    source_path = tmp_path / "mapping.xlsx"
    source_path.write_bytes(b"not a pdf")
    service = build_service(tmp_path)

    with pytest.raises(ValueError, match="Unsupported file extension"):
        service.store(source_path)


def test_store_rejects_oversized_file(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"x" * 5)
    service = build_service(tmp_path, max_upload_size_bytes=4)

    with pytest.raises(ValueError, match="exceeds maximum"):
        service.store(source_path)


def test_store_rejects_missing_file(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    with pytest.raises(FileNotFoundError, match="Source file does not exist"):
        service.store(tmp_path / "missing.pdf")
