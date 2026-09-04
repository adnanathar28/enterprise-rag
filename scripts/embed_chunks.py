from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from brd_knowledge.core.config import get_settings
from brd_knowledge.database.session import SessionLocal
from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.schemas.chunk import Chunk
from brd_knowledge.services.chunk_embedding_service import ChunkEmbeddingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed an existing chunks.json and synchronize it with PostgreSQL/pgvector."
    )
    parser.add_argument("chunks_path", type=Path, help="Path to an existing chunks.json.")
    return parser.parse_args()


def load_chunks(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file does not exist: {path}")
    return TypeAdapter(list[Chunk]).validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.chunks_path)
    if not chunks:
        raise ValueError("chunks.json is empty; cannot infer the document_id.")
    document_ids = {chunk.document_id for chunk in chunks}
    if len(document_ids) != 1:
        raise ValueError("All chunks must belong to exactly one document.")

    settings = get_settings()
    provider = GteModernBertEmbeddingProvider(
        model_name=settings.embedding_model_name,
        model_revision=settings.embedding_model_revision,
        dimension=settings.embedding_dimension,
        max_sequence_length=settings.embedding_max_sequence_length,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        preprocessing_version=settings.embedding_preprocessing_version,
    )
    with SessionLocal() as session:
        result = ChunkEmbeddingService(session, provider).synchronize(
            document_ids.pop(), chunks
        )

    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
