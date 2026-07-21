from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".pdf", ".docx"}
REQUIREMENT_ID_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){2,}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect raw Docling output for a PDF or DOCX BRD."
    )
    parser.add_argument("document_path", type=Path, help="Path to an approved BRD PDF or DOCX.")
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


def safe_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {str(key): safe_model_dump(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [safe_model_dump(item) for item in value]
    return value


def json_default(value: Any) -> str:
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def preview_text(text: str | None, limit: int = 220) -> str | None:
    if text is None:
        return None
    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    return f"{compact_text[: limit - 3]}..."


def get_label_name(item: Any) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return type(item).__name__
    return getattr(label, "value", str(label))


def get_text(item: Any) -> str | None:
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text
    return None


def extract_bbox(prov_item: Any) -> dict[str, Any] | None:
    bbox = getattr(prov_item, "bbox", None)
    if bbox is None:
        return None
    return safe_model_dump(bbox)


def extract_provenance(item: Any) -> list[dict[str, Any]]:
    provenance = []
    for prov_item in getattr(item, "prov", []) or []:
        provenance.append(
            {
                "page_no": getattr(prov_item, "page_no", None),
                "bbox": extract_bbox(prov_item),
                "charspan": safe_model_dump(getattr(prov_item, "charspan", None)),
            }
        )
    return provenance


def summarize_item(item: Any, index: int, level: int | None = None) -> dict[str, Any]:
    item_summary = {
        "index": index,
        "type": type(item).__name__,
        "label": get_label_name(item),
        "self_ref": getattr(item, "self_ref", None),
        "level": level,
        "text_preview": preview_text(get_text(item)),
        "provenance": extract_provenance(item),
    }
    return {key: value for key, value in item_summary.items() if value is not None}


def summarize_pages(docling_document: Any) -> list[dict[str, Any]]:
    pages = getattr(docling_document, "pages", {}) or {}
    page_summaries = []
    for page_no, page in sorted(pages.items()):
        page_summaries.append(
            {
                "page_no": page_no,
                "size": safe_model_dump(getattr(page, "size", None)),
                "image_present": getattr(page, "image", None) is not None,
            }
        )
    return page_summaries


def summarize_tables(docling_document: Any, limit: int = 10) -> list[dict[str, Any]]:
    table_summaries = []
    for index, table in enumerate(getattr(docling_document, "tables", []) or []):
        table_data = getattr(table, "data", None)
        table_cells = getattr(table_data, "table_cells", []) if table_data is not None else []
        grid = getattr(table_data, "grid", []) if table_data is not None else []
        table_summaries.append(
            {
                **summarize_item(table, index),
                "row_count": len(grid),
                "cell_count": len(table_cells),
                "sample_cells": [
                    {
                        "text": preview_text(getattr(cell, "text", None), limit=120),
                        "row_span": getattr(cell, "row_span", None),
                        "col_span": getattr(cell, "col_span", None),
                        "start_row_offset_idx": getattr(cell, "start_row_offset_idx", None),
                        "end_row_offset_idx": getattr(cell, "end_row_offset_idx", None),
                        "start_col_offset_idx": getattr(cell, "start_col_offset_idx", None),
                        "end_col_offset_idx": getattr(cell, "end_col_offset_idx", None),
                        "bbox": safe_model_dump(getattr(cell, "bbox", None)),
                    }
                    for cell in table_cells[:5]
                ],
            }
        )
        if len(table_summaries) >= limit:
            break
    return table_summaries


def summarize_pictures(docling_document: Any, limit: int = 10) -> list[dict[str, Any]]:
    pictures = []
    for index, picture in enumerate(getattr(docling_document, "pictures", []) or []):
        pictures.append(summarize_item(picture, index))
        if len(pictures) >= limit:
            break
    return pictures


def inspect_iterated_items(
    docling_document: Any,
) -> tuple[Counter[str], list[dict[str, Any]], list[dict[str, Any]]]:
    labels: Counter[str] = Counter()
    samples = []
    gaps = []

    for index, (item, level) in enumerate(docling_document.iterate_items(with_groups=True)):
        label = get_label_name(item)
        labels[label] += 1
        provenance = extract_provenance(item)

        has_text = get_text(item) is not None
        has_bbox = any(prov.get("bbox") is not None for prov in provenance)
        has_page = any(prov.get("page_no") is not None for prov in provenance)
        if has_text and not has_page:
            gaps.append({"index": index, "label": label, "issue": "text item has no page number"})
        if has_text and not has_bbox:
            gaps.append({"index": index, "label": label, "issue": "text item has no bounding box"})

        if len(samples) < 25 and (has_text or label.lower() in {"table", "picture"}):
            samples.append(summarize_item(item, index, level))

    return labels, samples, gaps


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "Docling Inspection Report",
        "",
        f"Source: {summary['source']['path']}",
        f"Docling version: {summary['parser']['docling_version']}",
        f"Conversion status: {summary['conversion']['status']}",
        f"Import seconds: {summary['timings']['docling_import_seconds']}",
        f"Conversion seconds: {summary['timings']['conversion_seconds']}",
        "",
        "Structure",
        f"- Top-level result type: {summary['conversion']['result_type']}",
        f"- Document type: {summary['document']['document_type']}",
        f"- Pages: {summary['counts']['pages']}",
        f"- Text items: {summary['counts']['texts']}",
        f"- Tables: {summary['counts']['tables']}",
        f"- Pictures: {summary['counts']['pictures']}",
        "",
        "Item Labels",
    ]

    for label, count in summary["counts"]["labels"].items():
        lines.append(f"- {label}: {count}")

    lines.extend(["", "Requirement ID Diagnostics"])
    lines.append(f"- Unique IDs found: {summary['requirement_id_diagnostics']['unique_count']}")
    for requirement_id, count in summary["requirement_id_diagnostics"]["counts"].items():
        lines.append(f"- {requirement_id}: {count}")

    lines.extend(["", "Extraction Gaps"])
    if summary["gaps"]:
        for gap in summary["gaps"][:50]:
            lines.append(f"- Item {gap['index']} ({gap['label']}): {gap['issue']}")
    else:
        lines.append("- No missing page/bounding-box gaps detected in sampled text items.")

    lines.extend(["", "Sample Items"])
    for item in summary["samples"]["items"]:
        text = item.get("text_preview") or "<no text>"
        provenance = item.get("provenance", [])
        page_numbers = sorted(
            {
                prov["page_no"]
                for prov in provenance
                if isinstance(prov, dict) and prov.get("page_no") is not None
            }
        )
        lines.append(
            f"- #{item['index']} {item['label']} "
            f"level={item.get('level')} pages={page_numbers}: {text}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    document_path = validate_document_path(args.document_path)
    output_dir = Path("data/parsed") / document_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    import_started_at = time.perf_counter()
    from docling.document_converter import DocumentConverter

    import_seconds = round(time.perf_counter() - import_started_at, 3)
    docling_version = metadata.version("docling")

    conversion_started_at = time.perf_counter()
    result = DocumentConverter().convert(document_path)
    conversion_seconds = round(time.perf_counter() - conversion_started_at, 3)
    docling_document = result.document

    raw_docling = docling_document.export_to_dict()
    markdown = docling_document.export_to_markdown(
        page_break_placeholder="\n\n<!-- page break -->\n\n"
    )
    labels, item_samples, gaps = inspect_iterated_items(docling_document)
    requirement_id_counts = Counter(REQUIREMENT_ID_PATTERN.findall(markdown))

    summary = {
        "source": {
            "path": str(document_path),
            "filename": document_path.name,
            "suffix": document_path.suffix.lower(),
            "size_bytes": document_path.stat().st_size,
        },
        "parser": {
            "name": "docling",
            "docling_version": docling_version,
        },
        "timings": {
            "docling_import_seconds": import_seconds,
            "conversion_seconds": conversion_seconds,
        },
        "conversion": {
            "result_type": type(result).__name__,
            "status": str(getattr(result, "status", None)),
            "errors": safe_model_dump(getattr(result, "errors", [])),
            "timings": safe_model_dump(getattr(result, "timings", None)),
            "confidence": safe_model_dump(getattr(result, "confidence", None)),
        },
        "document": {
            "document_type": type(docling_document).__name__,
            "name": getattr(docling_document, "name", None),
        },
        "counts": {
            "pages": len(getattr(docling_document, "pages", {}) or {}),
            "texts": len(getattr(docling_document, "texts", []) or []),
            "tables": len(getattr(docling_document, "tables", []) or []),
            "pictures": len(getattr(docling_document, "pictures", []) or []),
            "labels": dict(labels),
        },
        "pages": summarize_pages(docling_document),
        "samples": {
            "items": item_samples,
            "tables": summarize_tables(docling_document),
            "pictures": summarize_pictures(docling_document),
        },
        "requirement_id_diagnostics": {
            "pattern": REQUIREMENT_ID_PATTERN.pattern,
            "unique_count": len(requirement_id_counts),
            "counts": dict(sorted(requirement_id_counts.items())),
        },
        "gaps": gaps,
        "outputs": {
            "raw_json": str(output_dir / "docling_raw.json"),
            "markdown": str(output_dir / "docling_output.md"),
            "summary": str(output_dir / "inspection_summary.json"),
            "report": str(output_dir / "inspection_report.txt"),
        },
    }

    write_json(output_dir / "docling_raw.json", raw_docling)
    (output_dir / "docling_output.md").write_text(markdown, encoding="utf-8")
    write_json(output_dir / "inspection_summary.json", summary)
    (output_dir / "inspection_report.txt").write_text(build_report(summary), encoding="utf-8")

    print(f"Inspected: {document_path}")
    print(f"Output directory: {output_dir}")
    print(f"Conversion status: {summary['conversion']['status']}")
    print(
        "Counts: "
        f"{summary['counts']['pages']} pages, "
        f"{summary['counts']['texts']} text items, "
        f"{summary['counts']['tables']} tables, "
        f"{summary['counts']['pictures']} pictures"
    )


if __name__ == "__main__":
    main()
