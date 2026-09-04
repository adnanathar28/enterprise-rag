from __future__ import annotations

from typing import Protocol

from brd_knowledge.schemas.evaluation import (
    RetrievalEvalDataset,
    RetrievalEvalExample,
    RetrievalEvaluationReport,
    RetrievalExampleResult,
    RetrievalMetrics,
    RetrievedEvidence,
)
from brd_knowledge.schemas.retrieval import RetrievedChunk


class ChunkRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]: ...


class RetrievalEvaluator:
    def __init__(self, retriever: ChunkRetriever) -> None:
        self._retriever = retriever

    def evaluate(self, dataset: RetrievalEvalDataset) -> RetrievalEvaluationReport:
        results = [self._evaluate_example(example) for example in dataset.examples]
        count = len(results)
        metrics = RetrievalMetrics(
            example_count=count,
            recall_at_1=sum(result.hit_at_1 for result in results) / count,
            recall_at_3=sum(result.hit_at_3 for result in results) / count,
            recall_at_5=sum(result.hit_at_5 for result in results) / count,
            mrr=sum(
                1.0 / result.first_relevant_rank if result.first_relevant_rank else 0.0
                for result in results
            )
            / count,
        )
        return RetrievalEvaluationReport(
            dataset_name=dataset.name,
            metrics=metrics,
            results=results,
        )

    def _evaluate_example(self, example: RetrievalEvalExample) -> RetrievalExampleResult:
        retrieved_chunks = self._retriever.search(
            example.question,
            top_k=5,
            document_id=example.expected_document_id,
        )
        retrieved = [
            RetrievedEvidence(
                rank=chunk.rank,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                similarity=chunk.similarity,
                relevant=self._is_relevant(example, chunk),
            )
            for chunk in retrieved_chunks
        ]
        relevant_ranks = [item.rank for item in retrieved if item.relevant]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        return RetrievalExampleResult(
            question=example.question,
            expected_document_id=example.expected_document_id,
            relevant_chunk_ids=example.relevant_chunk_ids,
            section_page_targets=example.section_page_targets,
            notes=example.notes,
            retrieved=retrieved,
            first_relevant_rank=first_rank,
            hit_at_1=first_rank is not None and first_rank <= 1,
            hit_at_3=first_rank is not None and first_rank <= 3,
            hit_at_5=first_rank is not None and first_rank <= 5,
        )

    @staticmethod
    def _is_relevant(example: RetrievalEvalExample, chunk: RetrievedChunk) -> bool:
        if chunk.document_id != example.expected_document_id:
            return False
        if example.relevant_chunk_ids:
            return chunk.chunk_id in example.relevant_chunk_ids
        return any(
            (not target.section_path or chunk.section_path == target.section_path)
            and (
                not target.pages
                or any(chunk.page_start <= page <= chunk.page_end for page in target.pages)
            )
            for target in example.section_page_targets
        )
