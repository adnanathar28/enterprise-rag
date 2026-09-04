from __future__ import annotations

import argparse
import json
from pathlib import Path

from brd_knowledge.core.config import get_settings
from brd_knowledge.database.session import SessionLocal
from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.evaluation import RetrievalComparisonEvaluator
from brd_knowledge.retrieval import (
    DenseExperimentRetriever,
    HybridRrfRetriever,
    PgVectorRetriever,
    PostgresLexicalRetriever,
)
from brd_knowledge.schemas.evaluation import RetrievalComparisonReport, RetrievalEvalDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact dense chunk retrieval.")
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--output", type=Path, help="Optionally save the complete report as JSON.")
    return parser.parse_args()


def load_dataset(path: Path) -> RetrievalEvalDataset:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
    return RetrievalEvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def render_report(report: RetrievalComparisonReport) -> str:
    dataset_name = report.dense.dataset_name
    lines = [
        f"DATASET {dataset_name}",
        f"QUESTIONS {report.dense.metrics.example_count}",
        "",
        "STRATEGY       Recall@1  Recall@3  Recall@5  MRR",
    ]
    for name, strategy_report in (
        ("dense", report.dense),
        ("lexical", report.lexical),
        ("hybrid_rrf", report.hybrid),
    ):
        metrics = strategy_report.metrics
        lines.append(
            f"{name:<14} {metrics.recall_at_1:>8.3f}  {metrics.recall_at_3:>8.3f}  "
            f"{metrics.recall_at_5:>8.3f}  {metrics.mrr:>5.3f}"
        )
    lines.extend(["", "PER-QUESTION FIRST RELEVANT RANK"])
    for index, item in enumerate(report.per_question, 1):
        dense_rank = item.dense_first_relevant_rank or "NONE"
        lexical_rank = item.lexical_first_relevant_rank or "NONE"
        hybrid_rank = item.hybrid_first_relevant_rank or "NONE"
        lines.append(
            f"[{index}] dense={dense_rank} lexical={lexical_rank} hybrid={hybrid_rank} "
            f"question={item.question}"
        )
    lines.extend(["", "RECOVERED DENSE TOP-5 FAILURES"])
    lines.extend(f"- {question}" for question in report.recovered_dense_failures)
    if not report.recovered_dense_failures:
        lines.append("- NONE")
    lines.extend(["", "DENSE TOP-5 SUCCESSES REGRESSED BY HYBRID"])
    lines.extend(f"- {question}" for question in report.dense_success_regressions)
    if not report.dense_success_regressions:
        lines.append("- NONE")
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
        dense = DenseExperimentRetriever(PgVectorRetriever(session, provider))
        lexical = PostgresLexicalRetriever(session)
        hybrid = HybridRrfRetriever(dense, lexical)
        report = RetrievalComparisonEvaluator(dense, lexical, hybrid).evaluate(dataset)
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
