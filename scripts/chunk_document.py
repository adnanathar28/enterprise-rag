from __future__ import annotations

import argparse
import json
from pathlib import Path

from brd_knowledge.chunking import StructureAwareChunker
from brd_knowledge.schemas.chunk import Chunk
from brd_knowledge.schemas.document import ParsedDocument


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create inspectable structure-aware chunks from ParsedDocument JSON."
    )
    parser.add_argument("parsed_document_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: chunks.json beside parsed_document.json).",
    )
    parser.add_argument("--max-characters", type=int, default=3000)
    parser.add_argument("--preview-characters", type=int, default=240)
    return parser.parse_args()


def load_parsed_document(path: Path) -> ParsedDocument:
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Parsed document JSON does not exist: {resolved_path}")
    return ParsedDocument.model_validate_json(resolved_path.read_text(encoding="utf-8"))


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks], indent=2),
        encoding="utf-8",
    )


def render_chunks(chunks: list[Chunk], preview_characters: int = 240) -> str:
    lines = [f"CHUNKS {len(chunks)}", ""]
    for chunk in chunks:
        preview = " ".join(chunk.text.split())
        if len(preview) > preview_characters:
            preview = f"{preview[: preview_characters - 3]}..."
        path = " > ".join(chunk.section_path) if chunk.section_path else "NONE"
        quality = " | ".join(chunk.quality_notes) if chunk.quality_notes else "NONE"
        lines.extend(
            [
                f"chunk_id={chunk.chunk_id}",
                f"     section={path}",
                f"     content_type={chunk.content_type}",
                f"     pages={chunk.page_start}-{chunk.page_end}",
                f"     source_block_ids={','.join(chunk.source_block_ids) or 'NONE'}",
                f"     source_table_ids={','.join(chunk.source_table_ids) or 'NONE'}",
                f"     quality={quality}",
                f"     preview={json.dumps(preview, ensure_ascii=False)}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    document = load_parsed_document(args.parsed_document_path)
    chunks = StructureAwareChunker(max_characters=args.max_characters).chunk(document)
    output_path = args.output or args.parsed_document_path.with_name("chunks.json")
    write_chunks(output_path, chunks)
    print(render_chunks(chunks, preview_characters=args.preview_characters))
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
