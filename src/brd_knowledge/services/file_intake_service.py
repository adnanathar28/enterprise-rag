from pathlib import Path
from shutil import copy2
from uuid import uuid4

from brd_knowledge.schemas.source_file import StoredSourceFile


class FileIntakeService:
    def __init__(
        self,
        storage_dir: Path,
        allowed_extensions: set[str],
        max_upload_size_bytes: int,
    ) -> None:
        self._storage_dir = storage_dir
        self._allowed_extensions = {extension.lower() for extension in allowed_extensions}
        self._max_upload_size_bytes = max_upload_size_bytes

    def store(self, source_path: Path, original_filename: str | None = None) -> StoredSourceFile:
        resolved_path = source_path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise ValueError(f"Source path is not a file: {resolved_path}")

        filename = original_filename or resolved_path.name
        extension = Path(filename).suffix.lower()
        self._validate_extension(extension)
        size_bytes = resolved_path.stat().st_size
        self._validate_size(size_bytes)

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        stored_filename = f"{uuid4().hex}{extension}"
        stored_path = self._storage_dir / stored_filename
        copy2(resolved_path, stored_path)

        return StoredSourceFile(
            original_filename=filename,
            stored_filename=stored_filename,
            stored_path=stored_path,
            size_bytes=size_bytes,
            extension=extension,
        )

    def _validate_extension(self, extension: str) -> None:
        if extension not in self._allowed_extensions:
            allowed = ", ".join(sorted(self._allowed_extensions))
            raise ValueError(
                f"Unsupported file extension '{extension}'. Expected one of: {allowed}"
            )

    def _validate_size(self, size_bytes: int) -> None:
        if size_bytes > self._max_upload_size_bytes:
            raise ValueError(
                f"File size {size_bytes} exceeds maximum of {self._max_upload_size_bytes} bytes."
            )
