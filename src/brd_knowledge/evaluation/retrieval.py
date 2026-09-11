from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from brd_knowledge.schemas.evaluation import (
    EvidenceLocator,
    EvidenceLocatorAudit,
    QueryStrategyComparison,
    RetrievalComparisonReport,
    RetrievalDatasetAudit,
    RetrievalEvalDataset,
    RetrievalEvalExample,
    RetrievalEvaluationReport,
    RetrievalExampleResult,
    RetrievalMetrics,
    RetrievedEvidence,
)
from brd_knowledge.schemas.experimental_retrieval import ExperimentalRetrievedChunk


class ChunkRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[ExperimentalRetrievedChunk]: ...


class EvidenceCandidate(Protocol):
    chunk_id: str
    document_id: str
    section_path: list[str]
    text: str
    page_start: int
    page_end: int
    source_block_ids: list[str]
    source_table_ids: list[str]


LEGACY_FACT_ID = "legacy_relevance"


def normalize_evidence_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def locator_matches(locator: EvidenceLocator, chunk: EvidenceCandidate) -> bool:
    if locator.source_block_id and locator.source_block_id not in chunk.source_block_ids:
        return False
    if locator.source_table_id and locator.source_table_id not in chunk.source_table_ids:
        return False
    if locator.section_path and chunk.section_path != locator.section_path:
        return False
    if locator.pages and not any(
        chunk.page_start <= page <= chunk.page_end for page in locator.pages
    ):
        return False
    normalized_text = normalize_evidence_text(chunk.text)
    return all(
        normalize_evidence_text(anchor) in normalized_text
        for anchor in locator.text_anchors
    )


def required_fact_ids(example: RetrievalEvalExample) -> list[str]:
    if example.required_facts:
        return [fact.fact_id for fact in example.required_facts]
    return [LEGACY_FACT_ID]


def matched_fact_ids(
    example: RetrievalEvalExample,
    chunk: EvidenceCandidate,
) -> list[str]:
    if chunk.document_id != example.expected_document_id:
        return []
    if example.required_facts:
        return [
            fact.fact_id
            for fact in example.required_facts
            if any(locator_matches(locator, chunk) for locator in fact.acceptable_evidence)
        ]
    if example.relevant_chunk_ids:
        return (
            [LEGACY_FACT_ID]
            if chunk.chunk_id in example.relevant_chunk_ids
            else []
        )
    matches_target = any(
        (not target.section_path or chunk.section_path == target.section_path)
        and (
            not target.pages
            or any(chunk.page_start <= page <= chunk.page_end for page in target.pages)
        )
        for target in example.section_page_targets
    )
    return [LEGACY_FACT_ID] if matches_target else []


def covered_fact_ids(
    example: RetrievalEvalExample,
    chunks: Sequence[EvidenceCandidate],
    *,
    top_k: int,
) -> list[str]:
    matched = {
        fact_id
        for chunk in chunks[:top_k]
        for fact_id in matched_fact_ids(example, chunk)
    }
    return [fact_id for fact_id in required_fact_ids(example) if fact_id in matched]


def first_relevant_rank(
    example: RetrievalEvalExample,
    chunks: Sequence[EvidenceCandidate],
) -> int | None:
    return next(
        (
            rank
            for rank, chunk in enumerate(chunks, start=1)
            if matched_fact_ids(example, chunk)
        ),
        None,
    )


@dataclass(frozen=True)
class RankingEvaluation:
    first_relevant_rank: int | None
    first_rank_by_fact: dict[str, int | None]
    covered_fact_ids_at_1: list[str]
    covered_fact_ids_at_3: list[str]
    covered_fact_ids_at_5: list[str]
    required_fact_count: int

    def covered_at(self, top_k: int) -> list[str]:
        return {
            1: self.covered_fact_ids_at_1,
            3: self.covered_fact_ids_at_3,
            5: self.covered_fact_ids_at_5,
        }[top_k]

    def any_evidence_at(self, top_k: int) -> bool:
        return bool(self.covered_at(top_k))

    def full_coverage_at(self, top_k: int) -> bool:
        return len(self.covered_at(top_k)) == self.required_fact_count

    def fact_coverage_at(self, top_k: int) -> float:
        return len(self.covered_at(top_k)) / self.required_fact_count


def evaluate_ranking(
    example: RetrievalEvalExample,
    chunks: Sequence[EvidenceCandidate],
) -> RankingEvaluation:
    coverage = {
        top_k: covered_fact_ids(example, chunks, top_k=top_k)
        for top_k in (1, 3, 5)
    }
    fact_ranks = {
        fact_id: next(
            (
                rank
                for rank, chunk in enumerate(chunks, start=1)
                if fact_id in matched_fact_ids(example, chunk)
            ),
            None,
        )
        for fact_id in required_fact_ids(example)
    }
    found_ranks = [rank for rank in fact_ranks.values() if rank is not None]
    return RankingEvaluation(
        first_relevant_rank=min(found_ranks) if found_ranks else None,
        first_rank_by_fact=fact_ranks,
        covered_fact_ids_at_1=coverage[1],
        covered_fact_ids_at_3=coverage[3],
        covered_fact_ids_at_5=coverage[5],
        required_fact_count=len(required_fact_ids(example)),
    )


def aggregate_metrics(evaluations: Sequence[RankingEvaluation]) -> RetrievalMetrics:
    count = len(evaluations)
    if count == 0:
        raise ValueError("Cannot aggregate an empty retrieval evaluation.")

    def average(values: Sequence[float | bool]) -> float:
        return sum(values) / count

    any_at = {
        top_k: average([item.any_evidence_at(top_k) for item in evaluations])
        for top_k in (1, 3, 5)
    }
    full_at = {
        top_k: average([item.full_coverage_at(top_k) for item in evaluations])
        for top_k in (1, 3, 5)
    }
    fact_at = {
        top_k: average([item.fact_coverage_at(top_k) for item in evaluations])
        for top_k in (1, 3, 5)
    }
    mrr_at_5 = average(
        [
            1.0 / item.first_relevant_rank
            if item.first_relevant_rank and item.first_relevant_rank <= 5
            else 0.0
            for item in evaluations
        ]
    )
    return RetrievalMetrics(
        example_count=count,
        recall_at_1=any_at[1],
        recall_at_3=any_at[3],
        recall_at_5=any_at[5],
        mrr=mrr_at_5,
        any_evidence_at_1=any_at[1],
        any_evidence_at_3=any_at[3],
        any_evidence_at_5=any_at[5],
        full_coverage_at_1=full_at[1],
        full_coverage_at_3=full_at[3],
        full_coverage_at_5=full_at[5],
        mean_fact_coverage_at_1=fact_at[1],
        mean_fact_coverage_at_3=fact_at[3],
        mean_fact_coverage_at_5=fact_at[5],
        mrr_at_5=mrr_at_5,
    )


def audit_dataset_locators(
    dataset: RetrievalEvalDataset,
    chunks: Sequence[EvidenceCandidate],
    *,
    max_expected_matches: int = 3,
) -> RetrievalDatasetAudit:
    if max_expected_matches <= 0:
        raise ValueError("max_expected_matches must be greater than zero.")
    locators: list[EvidenceLocatorAudit] = []
    for example in dataset.examples:
        for fact in example.required_facts:
            for locator_index, locator in enumerate(fact.acceptable_evidence, start=1):
                matches = [
                    chunk.chunk_id
                    for chunk in chunks
                    if chunk.document_id == example.expected_document_id
                    and locator_matches(locator, chunk)
                ]
                status: Literal["ok", "zero_matches", "unexpectedly_many"]
                if not matches:
                    status = "zero_matches"
                elif len(matches) > max_expected_matches:
                    status = "unexpectedly_many"
                else:
                    status = "ok"
                locators.append(
                    EvidenceLocatorAudit(
                        question=example.question,
                        fact_id=fact.fact_id,
                        locator_index=locator_index,
                        match_count=len(matches),
                        matched_chunk_ids=matches,
                        status=status,
                    )
                )
    return RetrievalDatasetAudit(
        locator_count=len(locators),
        zero_match_count=sum(item.status == "zero_matches" for item in locators),
        unexpectedly_many_count=sum(
            item.status == "unexpectedly_many" for item in locators
        ),
        max_expected_matches=max_expected_matches,
        locators=locators,
    )


class RetrievalEvaluator:
    def __init__(self, retriever: ChunkRetriever) -> None:
        self._retriever = retriever

    def evaluate(self, dataset: RetrievalEvalDataset) -> RetrievalEvaluationReport:
        results = [self._evaluate_example(example) for example in dataset.examples]
        evaluations = [
            RankingEvaluation(
                first_relevant_rank=result.first_relevant_rank,
                first_rank_by_fact=result.first_rank_by_fact,
                covered_fact_ids_at_1=result.covered_fact_ids_at_1,
                covered_fact_ids_at_3=result.covered_fact_ids_at_3,
                covered_fact_ids_at_5=result.covered_fact_ids_at_5,
                required_fact_count=len(required_fact_ids(example)),
            )
            for example, result in zip(dataset.examples, results, strict=True)
        ]
        metrics = aggregate_metrics(evaluations)
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
                score=chunk.score,
                strategy=chunk.strategy,
                relevant=bool(matched_fact_ids(example, chunk)),
                matched_fact_ids=matched_fact_ids(example, chunk),
            )
            for chunk in retrieved_chunks
        ]
        evaluation = evaluate_ranking(example, retrieved_chunks)

        return RetrievalExampleResult(
            question=example.question,
            expected_document_id=example.expected_document_id,
            relevant_chunk_ids=example.relevant_chunk_ids,
            section_page_targets=example.section_page_targets,
            required_facts=example.required_facts,
            notes=example.notes,
            retrieved=retrieved,
            first_relevant_rank=evaluation.first_relevant_rank,
            first_rank_by_fact=evaluation.first_rank_by_fact,
            hit_at_1=evaluation.any_evidence_at(1),
            hit_at_3=evaluation.any_evidence_at(3),
            hit_at_5=evaluation.any_evidence_at(5),
            covered_fact_ids_at_1=evaluation.covered_fact_ids_at_1,
            covered_fact_ids_at_3=evaluation.covered_fact_ids_at_3,
            covered_fact_ids_at_5=evaluation.covered_fact_ids_at_5,
            full_coverage_at_1=evaluation.full_coverage_at(1),
            full_coverage_at_3=evaluation.full_coverage_at(3),
            full_coverage_at_5=evaluation.full_coverage_at(5),
            mean_fact_coverage_at_1=evaluation.fact_coverage_at(1),
            mean_fact_coverage_at_3=evaluation.fact_coverage_at(3),
            mean_fact_coverage_at_5=evaluation.fact_coverage_at(5),
        )

    @staticmethod
    def _is_relevant(
        example: RetrievalEvalExample,
        chunk: ExperimentalRetrievedChunk,
    ) -> bool:
        return bool(matched_fact_ids(example, chunk))


class RetrievalComparisonEvaluator:
    def __init__(
        self,
        dense: ChunkRetriever,
        lexical: ChunkRetriever,
        hybrid: ChunkRetriever,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._hybrid = hybrid

    def evaluate(self, dataset: RetrievalEvalDataset) -> RetrievalComparisonReport:
        dense = RetrievalEvaluator(self._dense).evaluate(dataset)
        lexical = RetrievalEvaluator(self._lexical).evaluate(dataset)
        hybrid = RetrievalEvaluator(self._hybrid).evaluate(dataset)
        per_question = [
            QueryStrategyComparison(
                question=dense_result.question,
                dense_first_relevant_rank=dense_result.first_relevant_rank,
                lexical_first_relevant_rank=lexical_result.first_relevant_rank,
                hybrid_first_relevant_rank=hybrid_result.first_relevant_rank,
            )
            for dense_result, lexical_result, hybrid_result in zip(
                dense.results,
                lexical.results,
                hybrid.results,
                strict=True,
            )
        ]
        recovered = [
            item.question
            for item, dense_result, hybrid_result in zip(
                per_question,
                dense.results,
                hybrid.results,
                strict=True,
            )
            if not dense_result.full_coverage_at_5 and hybrid_result.full_coverage_at_5
        ]
        regressions = [
            item.question
            for item, dense_result, hybrid_result in zip(
                per_question,
                dense.results,
                hybrid.results,
                strict=True,
            )
            if dense_result.full_coverage_at_5 and not hybrid_result.full_coverage_at_5
        ]
        return RetrievalComparisonReport(
            dense=dense,
            lexical=lexical,
            hybrid=hybrid,
            per_question=per_question,
            recovered_dense_failures=recovered,
            dense_success_regressions=regressions,
        )
