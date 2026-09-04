from __future__ import annotations

import argparse

from brd_knowledge.core.config import get_settings
from brd_knowledge.database.session import SessionLocal
from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.retrieval import PgVectorRetriever
from brd_knowledge.schemas.retrieval import RetrievedChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search persisted chunks with exact cosine search."
    )
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--document-id", help="Optionally restrict results to one document.")
    parser.add_argument("--preview-characters", type=int, default=300)
    return parser.parse_args()


def render_results(results: list[RetrievedChunk], preview_characters: int = 300) -> str:
    lines = [f"RESULTS {len(results)}", ""]
    for result in results:
        preview = " ".join(result.text.split())
        if len(preview) > preview_characters:
            preview = f"{preview[: preview_characters - 3]}..."
        section = " > ".join(result.section_path) or "NONE"
        lines.extend(
            [
                f"[{result.rank}] similarity={result.similarity:.6f}",
                f"    chunk_id={result.chunk_id}",
                f"    document_id={result.document_id}",
                f"    section={section}",
                f"    content_type={result.content_type}",
                f"    pages={result.page_start}-{result.page_end}",
                f"    source_block_ids={','.join(result.source_block_ids) or 'NONE'}",
                f"    source_table_ids={','.join(result.source_table_ids) or 'NONE'}",
                f'    preview="{preview}"',
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.preview_characters <= 0:
        raise ValueError("preview-characters must be greater than zero.")
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
        results = PgVectorRetriever(session, provider).search(
            args.query,
            top_k=args.top_k,
            document_id=args.document_id,
        )
    print(render_results(results, args.preview_characters))


if __name__ == "__main__":
    main()
