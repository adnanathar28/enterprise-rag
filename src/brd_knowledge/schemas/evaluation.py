from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from brd_knowledge.schemas.experimental_retrieval import ExperimentalStrategy


class SectionPageTarget(BaseModel):
    section_path: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_section_or_page(self) -> SectionPageTarget:
        if not self.section_path and not self.pages:
            raise ValueError("A section/page target must contain a section path or page.")
        return self


class EvidenceLocator(BaseModel):
    source_block_id: str | None = None
    source_table_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    text_anchors: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def require_scoped_content_identity(self) -> EvidenceLocator:
        if any(not anchor.strip() for anchor in self.text_anchors):
            raise ValueError("Evidence text anchors must not be blank.")
        if not (
            self.source_block_id
            or self.source_table_id
            or self.section_path
            or self.pages
        ):
            raise ValueError(
                "An evidence locator must scope its text anchors by source, section, or page."
            )
        return self


class RequiredFact(BaseModel):
    fact_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    acceptable_evidence: list[EvidenceLocator] = Field(min_length=1)


class RetrievalEvalExample(BaseModel):
    question: str = Field(min_length=1)
    expected_document_id: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    section_page_targets: list[SectionPageTarget] = Field(default_factory=list)
    required_facts: list[RequiredFact] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def require_relevant_evidence(self) -> RetrievalEvalExample:
        has_v1_labels = bool(self.relevant_chunk_ids or self.section_page_targets)
        if has_v1_labels == bool(self.required_facts):
            raise ValueError(
                "An example must define either v1 relevance labels or v2 required facts."
            )
        fact_ids = [fact.fact_id for fact in self.required_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("Required fact IDs must be unique within an example.")
        return self


class RetrievalEvalDataset(BaseModel):
    name: str = Field(min_length=1)
    schema_version: int = Field(default=1, ge=1, le=2)
    examples: list[RetrievalEvalExample] = Field(min_length=1)

    @model_validator(mode="after")
    def require_version_consistent_examples(self) -> RetrievalEvalDataset:
        if self.schema_version == 2 and any(
            not example.required_facts for example in self.examples
        ):
            raise ValueError("Schema v2 datasets must use required facts for every example.")
        if self.schema_version == 1 and any(
            example.required_facts for example in self.examples
        ):
            raise ValueError("Schema v1 datasets must use legacy relevance labels.")
        return self


class RetrievedEvidence(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str
    document_id: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    score: float
    strategy: ExperimentalStrategy
    relevant: bool
    matched_fact_ids: list[str] = Field(default_factory=list)


class RetrievalExampleResult(BaseModel):
    question: str
    expected_document_id: str
    relevant_chunk_ids: list[str]
    section_page_targets: list[SectionPageTarget]
    required_facts: list[RequiredFact] = Field(default_factory=list)
    notes: str | None = None
    retrieved: list[RetrievedEvidence]
    first_relevant_rank: int | None
    first_rank_by_fact: dict[str, int | None] = Field(default_factory=dict)
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    covered_fact_ids_at_1: list[str] = Field(default_factory=list)
    covered_fact_ids_at_3: list[str] = Field(default_factory=list)
    covered_fact_ids_at_5: list[str] = Field(default_factory=list)
    full_coverage_at_1: bool
    full_coverage_at_3: bool
    full_coverage_at_5: bool
    mean_fact_coverage_at_1: float
    mean_fact_coverage_at_3: float
    mean_fact_coverage_at_5: float


class RetrievalMetrics(BaseModel):
    example_count: int = Field(ge=0)
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    any_evidence_at_1: float
    any_evidence_at_3: float
    any_evidence_at_5: float
    full_coverage_at_1: float
    full_coverage_at_3: float
    full_coverage_at_5: float
    mean_fact_coverage_at_1: float
    mean_fact_coverage_at_3: float
    mean_fact_coverage_at_5: float
    mrr_at_5: float


class RetrievalEvaluationReport(BaseModel):
    dataset_name: str
    metrics: RetrievalMetrics
    results: list[RetrievalExampleResult]


class EvidenceLocatorAudit(BaseModel):
    question: str
    fact_id: str
    locator_index: int = Field(ge=1)
    match_count: int = Field(ge=0)
    matched_chunk_ids: list[str] = Field(default_factory=list)
    status: Literal["ok", "zero_matches", "unexpectedly_many"]


class RetrievalDatasetAudit(BaseModel):
    locator_count: int = Field(ge=0)
    zero_match_count: int = Field(ge=0)
    unexpectedly_many_count: int = Field(ge=0)
    max_expected_matches: int = Field(ge=1)
    locators: list[EvidenceLocatorAudit] = Field(default_factory=list)


class QueryStrategyComparison(BaseModel):
    question: str
    dense_first_relevant_rank: int | None
    lexical_first_relevant_rank: int | None
    hybrid_first_relevant_rank: int | None


class RetrievalComparisonReport(BaseModel):
    dense: RetrievalEvaluationReport
    lexical: RetrievalEvaluationReport
    hybrid: RetrievalEvaluationReport
    per_question: list[QueryStrategyComparison]
    recovered_dense_failures: list[str]
    dense_success_regressions: list[str]
