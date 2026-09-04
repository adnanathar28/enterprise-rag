from __future__ import annotations

import argparse
import json
from pathlib import Path

from brd_knowledge.core.config import get_settings
from brd_knowledge.database.session import SessionLocal
from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.evaluation import RetrievalEvaluator
from brd_knowledge.retrieval import PgVectorRetriever
from brd_knowledge.schemas.evaluation import RetrievalEvalDataset, RetrievalEvaluationReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact dense chunk retrieval.")
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--output", type=Path, help="Optionally save the complete report as JSON.")
    return parser.parse_args()


def load_dataset(path: Path) -> RetrievalEvalDataset:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
    return RetrievalEvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def render_report(report: RetrievalEvaluationReport) -> str:
    metrics = report.metrics
    lines = [
        f"DATASET {report.dataset_name}",
        f"QUESTIONS {metrics.example_count}",
        f"Recall@1 {metrics.recall_at_1:.3f}",
        f"Recall@3 {metrics.recall_at_3:.3f}",
        f"Recall@5 {metrics.recall_at_5:.3f}",
        f"MRR {metrics.mrr:.3f}",
        "",
    ]
    for index, result in enumerate(report.results, 1):
        status = "PASS" if result.hit_at_5 else "FAIL"
        expected = ", ".join(result.relevant_chunk_ids) or "section/page target"
        lines.extend(
            [
                f"[{index}] {status} first_relevant_rank={result.first_relevant_rank or 'NONE'}",
                f"    question={result.question}",
                f"    expected={expected}",
            ]
        )
        for item in result.retrieved:
            marker = "RELEVANT" if item.relevant else "-"
            section = " > ".join(item.section_path) or "NONE"
            lines.append(
                f"    rank={item.rank} {marker} score={item.similarity:.6f} "
                f"chunk={item.chunk_id} pages={item.page_start}-{item.page_end} "
                f"section={section}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_path)
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
        report = RetrievalEvaluator(PgVectorRetriever(session, provider)).evaluate(dataset)
    print(render_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
