from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from sqlalchemy import select

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.database.session import SessionLocal
from brd_knowledge.evaluation.retrieval import audit_dataset_locators
from brd_knowledge.schemas.chunk import ChunkContentType
from brd_knowledge.schemas.evaluation import RetrievalDatasetAudit, RetrievalEvalDataset
from brd_knowledge.schemas.experimental_retrieval import ExperimentalRetrievedChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit stable retrieval-evaluation locators against stored chunks."
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--max-matches", type=int, default=3)
    return parser.parse_args()


def load_dataset(path: Path) -> RetrievalEvalDataset:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
    return RetrievalEvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def stored_chunk(row: ChunkEmbedding, rank: int) -> ExperimentalRetrievedChunk:
    return ExperimentalRetrievedChunk(
        rank=rank,
        score=0,
        strategy="dense",
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        section_id=row.section_id,
        section_title=row.section_title,
        section_path=row.section_path,
        content_type=cast(ChunkContentType, row.content_type),
        text=row.text,
        page_start=row.page_start,
        page_end=row.page_end,
        source_block_ids=row.source_block_ids,
        source_table_ids=row.source_table_ids,
        provenance=row.provenance,
        quality_notes=row.quality_notes,
    )


def render_audit(audit: RetrievalDatasetAudit) -> str:
    lines = [
        f"LOCATORS {audit.locator_count}",
        f"ZERO_MATCHES {audit.zero_match_count}",
        f"UNEXPECTEDLY_MANY {audit.unexpectedly_many_count}",
    ]
    for item in audit.locators:
        if item.status == "ok":
            continue
        lines.append(
            f"{item.status.upper()} fact={item.fact_id} locator={item.locator_index} "
            f"matches={item.match_count} question={item.question}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_path)
    document_ids = {example.expected_document_id for example in dataset.examples}
    with SessionLocal() as session:
        rows = session.scalars(
            select(ChunkEmbedding).where(ChunkEmbedding.document_id.in_(document_ids))
        ).all()
    chunks = [stored_chunk(row, rank) for rank, row in enumerate(rows, start=1)]
    audit = audit_dataset_locators(
        dataset,
        chunks,
        max_expected_matches=args.max_matches,
    )
    print(render_audit(audit))
    if audit.zero_match_count or audit.unexpectedly_many_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
