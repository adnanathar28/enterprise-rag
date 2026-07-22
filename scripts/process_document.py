from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brd_knowledge.parsing.docling_parser import DoclingDocumentParser

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a BRD into the internal ParsedDocument JSON schema."
    )
    parser.add_argument("document_path", type=Path, help="Path to an approved BRD PDF or DOCX.")
    parser.add_argument(
        "--page-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional inclusive 1-based page range to process.",
    )
    return parser.parse_args()


def validate_document_path(document_path: Path) -> Path:
    resolved_path = document_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Document does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Document path is not a file: {resolved_path}")
    if resolved_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(
            f"Unsupported file type '{resolved_path.suffix}'. Expected one of: {supported}"
        )
    return resolved_path


def validate_page_range(page_range: list[int] | None) -> tuple[int, int] | None:
    if page_range is None:
        return None

    start, end = page_range
    if start < 1 or end < 1:
        raise ValueError("Page range values must be positive 1-based page numbers.")
    if start > end:
        raise ValueError(f"Invalid page range: start page {start} is after end page {end}.")
    return start, end


def build_output_dir(document_path: Path, page_range: tuple[int, int] | None) -> Path:
    output_name = document_path.stem
    if page_range is not None:
        output_name = f"{output_name}_pages_{page_range[0]}_{page_range[1]}"
    return Path("data/outputs") / output_name


def json_default(value: Any) -> str:
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    print("Validating document...")
    document_path = validate_document_path(args.document_path)
    page_range = validate_page_range(args.page_range)

    output_dir = build_output_dir(document_path, page_range)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing with Docling adapter...")
    parsed_document = DoclingDocumentParser(page_range=page_range).parse(document_path)

    output_path = output_dir / "parsed_document.json"
    print("Writing normalized output...")
    write_json(output_path, parsed_document.model_dump(mode="json"))

    parse_status = (
        parsed_document.parser_metadata.parse_status
        if parsed_document.parser_metadata is not None
        else None
    )
    print(f"Processed: {document_path}")
    print(f"Output file: {output_path}")
    print(f"Parse status: {parse_status}")
    print(
        "Counts: "
        f"{len(parsed_document.pages)} pages, "
        f"{len(parsed_document.blocks)} blocks, "
        f"{len(parsed_document.tables)} tables, "
        f"{len(parsed_document.images)} images, "
        f"{len(parsed_document.sections)} sections, "
        f"{len(parsed_document.diagnostics)} diagnostics"
    )
    print("Done.")


if __name__ == "__main__":
    main()
