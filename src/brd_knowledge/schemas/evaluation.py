from __future__ import annotations

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


class RetrievalEvalExample(BaseModel):
    question: str = Field(min_length=1)
    expected_document_id: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    section_page_targets: list[SectionPageTarget] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def require_relevant_evidence(self) -> RetrievalEvalExample:
        if not self.relevant_chunk_ids and not self.section_page_targets:
            raise ValueError("An example must define relevant chunks or section/page targets.")
        return self


class RetrievalEvalDataset(BaseModel):
    name: str = Field(min_length=1)
    examples: list[RetrievalEvalExample] = Field(min_length=1)


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


class RetrievalExampleResult(BaseModel):
    question: str
    expected_document_id: str
    relevant_chunk_ids: list[str]
    section_page_targets: list[SectionPageTarget]
    notes: str | None = None
    retrieved: list[RetrievedEvidence]
    first_relevant_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool


class RetrievalMetrics(BaseModel):
    example_count: int = Field(ge=0)
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float


class RetrievalEvaluationReport(BaseModel):
    dataset_name: str
    metrics: RetrievalMetrics
    results: list[RetrievalExampleResult]


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
