from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from brd_knowledge.core.config import get_settings
from brd_knowledge.database.session import SessionLocal
from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.evaluation.retrieval import aggregate_metrics, evaluate_ranking
from brd_knowledge.retrieval.experimental_reranking import (
    MODEL_NAME,
    MODEL_REVISION,
    TransformersCrossEncoderScorer,
    rerank_chunks,
)
from brd_knowledge.retrieval.pgvector import PgVectorRetriever
from brd_knowledge.schemas.evaluation import RetrievalEvalDataset
from brd_knowledge.schemas.retrieval import RetrievedChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline comparison of dense retrieval and local cross-encoder reranking."
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_dataset(path: Path) -> RetrievalEvalDataset:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
    return RetrievalEvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def chunk_summary(chunk: RetrievedChunk, *, rank: int, score: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": score,
        "chunk_id": chunk.chunk_id,
        "section": " > ".join(chunk.section_path),
        "pages": [chunk.page_start, chunk.page_end],
        "content_type": chunk.content_type,
    }


def render_report(report: dict[str, Any]) -> str:
    dense = report["metrics"]["dense"]
    reranked = report["metrics"]["reranked"]
    lines = [
        f"DATASET {report['dataset_name']}",
        f"MODEL {report['reranker']['model_name']}@{report['reranker']['revision']}",
        f"MODEL_LOAD_SECONDS {report['latency']['model_load_seconds']:.6f}",
        "",
        "STRATEGY  Any@1  Any@3  Any@5  Full@1  Full@3  Full@5  MeanFact@5  MRR@5",
        f"dense     {dense['any_evidence_at_1']:.3f}  "
        f"{dense['any_evidence_at_3']:.3f}  {dense['any_evidence_at_5']:.3f}  "
        f"{dense['full_coverage_at_1']:.3f}   {dense['full_coverage_at_3']:.3f}   "
        f"{dense['full_coverage_at_5']:.3f}   "
        f"{dense['mean_fact_coverage_at_5']:.3f}       {dense['mrr_at_5']:.3f}",
        f"reranked  {reranked['any_evidence_at_1']:.3f}  "
        f"{reranked['any_evidence_at_3']:.3f}  {reranked['any_evidence_at_5']:.3f}  "
        f"{reranked['full_coverage_at_1']:.3f}   "
        f"{reranked['full_coverage_at_3']:.3f}   "
        f"{reranked['full_coverage_at_5']:.3f}   "
        f"{reranked['mean_fact_coverage_at_5']:.3f}       {reranked['mrr_at_5']:.3f}",
        "",
    ]
    for index, result in enumerate(report["results"], 1):
        lines.extend(
            [
                f"[{index}] {result['question']}",
                f"dense_first_relevant={result['dense_first_relevant_rank']} "
                f"reranked_first_relevant={result['reranked_first_relevant_rank']}",
                f"dense_seconds={result['dense_seconds']:.6f} "
                f"rerank_seconds={result['rerank_seconds']:.6f} "
                f"total_seconds={result['total_seconds']:.6f}",
                "dense_top_5="
                + ", ".join(item["chunk_id"] for item in result["dense_top_5"]),
                "reranked_top_5="
                + ", ".join(item["chunk_id"] for item in result["reranked_top_5"]),
                "gold_ranks=" + json.dumps(result["gold_ranks"], sort_keys=True),
                "dense_fact_ranks="
                + json.dumps(result["dense_first_rank_by_fact"], sort_keys=True),
                "reranked_fact_ranks="
                + json.dumps(result["reranked_first_rank_by_fact"], sort_keys=True),
                "dense_covered_facts_at_5="
                + json.dumps(result["dense_covered_fact_ids_at_5"]),
                "reranked_covered_facts_at_5="
                + json.dumps(result["reranked_covered_fact_ids_at_5"]),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.candidate_k < args.top_k:
        raise ValueError("candidate-k must be greater than or equal to top-k.")
    dataset = load_dataset(args.dataset_path)
    settings = get_settings()
    embedding_provider = GteModernBertEmbeddingProvider(
        model_name=settings.embedding_model_name,
        model_revision=settings.embedding_model_revision,
        dimension=settings.embedding_dimension,
        max_sequence_length=settings.embedding_max_sequence_length,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        preprocessing_version=settings.embedding_preprocessing_version,
    )
    scorer = TransformersCrossEncoderScorer(batch_size=args.batch_size, device=args.device)

    model_load_started = perf_counter()
    scorer._ensure_loaded()
    model_load_seconds = perf_counter() - model_load_started

    results: list[dict[str, Any]] = []
    dense_evaluations = []
    reranked_evaluations = []
    with SessionLocal() as session:
        retriever = PgVectorRetriever(session, embedding_provider)
        for example in dataset.examples:
            dense_started = perf_counter()
            candidates = retriever.search(
                example.question,
                top_k=args.candidate_k,
                document_id=example.expected_document_id,
            )
            dense_seconds = perf_counter() - dense_started

            rerank_started = perf_counter()
            reranked = rerank_chunks(
                example.question,
                candidates,
                scorer,
                top_k=len(candidates),
            )
            rerank_seconds = perf_counter() - rerank_started
            dense_top = candidates[: args.top_k]
            reranked_top = reranked[: args.top_k]
            reranked_chunks = [item.chunk for item in reranked]
            dense_evaluation = evaluate_ranking(example, candidates)
            reranked_evaluation = evaluate_ranking(example, reranked_chunks)
            dense_evaluations.append(dense_evaluation)
            reranked_evaluations.append(reranked_evaluation)

            candidate_ranks = {chunk.chunk_id: chunk.rank for chunk in candidates}
            reranked_ranks = {item.chunk.chunk_id: item.rank for item in reranked}
            results.append(
                {
                    "question": example.question,
                    "dense_first_relevant_rank": dense_evaluation.first_relevant_rank,
                    "reranked_first_relevant_rank": (
                        reranked_evaluation.first_relevant_rank
                    ),
                    "dense_first_rank_by_fact": dense_evaluation.first_rank_by_fact,
                    "reranked_first_rank_by_fact": (
                        reranked_evaluation.first_rank_by_fact
                    ),
                    "dense_covered_fact_ids_at_5": (
                        dense_evaluation.covered_fact_ids_at_5
                    ),
                    "reranked_covered_fact_ids_at_5": (
                        reranked_evaluation.covered_fact_ids_at_5
                    ),
                    "dense_seconds": dense_seconds,
                    "rerank_seconds": rerank_seconds,
                    "total_seconds": dense_seconds + rerank_seconds,
                    "dense_top_5": [
                        chunk_summary(chunk, rank=chunk.rank, score=chunk.similarity)
                        for chunk in dense_top
                    ],
                    "reranked_top_5": [
                        chunk_summary(
                            item.chunk,
                            rank=item.rank,
                            score=item.reranker_score,
                        )
                        for item in reranked_top
                    ],
                    "gold_ranks": {
                        chunk_id: {
                            "dense": candidate_ranks.get(chunk_id),
                            "reranked": reranked_ranks.get(chunk_id),
                        }
                        for chunk_id in example.relevant_chunk_ids
                    },
                }
            )

    report = {
        "dataset_name": dataset.name,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "reranker": {
            "model_name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "max_sequence_length": scorer.max_sequence_length,
            "batch_size": scorer.batch_size,
            "device": scorer.device,
            "scoring": "single classification logit, descending",
        },
        "latency": {
            "model_load_seconds": model_load_seconds,
            "dense_total_seconds": sum(item["dense_seconds"] for item in results),
            "rerank_total_seconds": sum(item["rerank_seconds"] for item in results),
            "combined_total_seconds": sum(item["total_seconds"] for item in results),
        },
        "metrics": {
            "dense": aggregate_metrics(dense_evaluations).model_dump(mode="json"),
            "reranked": aggregate_metrics(reranked_evaluations).model_dump(mode="json"),
        },
        "results": results,
    }
    print(render_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
