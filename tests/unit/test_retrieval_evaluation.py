from __future__ import annotations

from brd_knowledge.evaluation import RetrievalComparisonEvaluator, RetrievalEvaluator
from brd_knowledge.schemas.evaluation import RetrievalEvalDataset, RetrievalEvalExample
from brd_knowledge.schemas.experimental_retrieval import ExperimentalRetrievedChunk


def result(chunk_id: str, rank: int, *, page: int = 1) -> ExperimentalRetrievedChunk:
    return ExperimentalRetrievedChunk(
        rank=rank,
        score=1 - rank / 10,
        strategy="dense",
        chunk_id=chunk_id,
        document_id="doc-1",
        section_path=["Requirements"],
        content_type="prose",
        text=f"Evidence {chunk_id}",
        page_start=page,
        page_end=page,
    )


class FakeRetriever:
    def __init__(self, results: dict[str, list[ExperimentalRetrievedChunk]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, str | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[ExperimentalRetrievedChunk]:
        self.calls.append((query, top_k, document_id))
        return self.results[query]


def example(question: str, relevant: list[str]) -> RetrievalEvalExample:
    return RetrievalEvalExample(
        question=question,
        expected_document_id="doc-1",
        relevant_chunk_ids=relevant,
    )


def test_metrics_multiple_labels_misses_and_short_result_lists() -> None:
    dataset = RetrievalEvalDataset(
        name="test",
        examples=[
            example("rank one", ["a"]),
            example("multiple acceptable", ["missing", "b"]),
            example("no relevant", ["wanted"]),
            example("short list", ["d"]),
        ],
    )
    retriever = FakeRetriever(
        {
            "rank one": [result("a", 1), result("x", 2)],
            "multiple acceptable": [result("x", 1), result("y", 2), result("b", 3)],
            "no relevant": [result("x", 1), result("y", 2)],
            "short list": [result("x", 1), result("d", 2)],
        }
    )

    report = RetrievalEvaluator(retriever).evaluate(dataset)

    assert report.metrics.recall_at_1 == 0.25
    assert report.metrics.recall_at_3 == 0.75
    assert report.metrics.recall_at_5 == 0.75
    assert report.metrics.mrr == (1 + 1 / 3 + 0 + 1 / 2) / 4
    assert [item.first_relevant_rank for item in report.results] == [1, 3, None, 2]
    assert all(call[1:] == (5, "doc-1") for call in retriever.calls)


def test_section_and_page_target_is_used_when_chunk_ids_are_not_supplied() -> None:
    dataset = RetrievalEvalDataset.model_validate(
        {
            "name": "target fallback",
            "examples": [
                {
                    "question": "target",
                    "expected_document_id": "doc-1",
                    "section_page_targets": [
                        {"section_path": ["Requirements"], "pages": [2]}
                    ],
                }
            ],
        }
    )
    retriever = FakeRetriever({"target": [result("other-id", 1, page=2)]})

    report = RetrievalEvaluator(retriever).evaluate(dataset)

    assert report.results[0].first_relevant_rank == 1
    assert report.results[0].retrieved[0].relevant is True


def test_evaluation_output_is_deterministic() -> None:
    dataset = RetrievalEvalDataset(
        name="stable",
        examples=[example("first", ["a"]), example("second", ["b"])],
    )
    retriever = FakeRetriever(
        {
            "first": [result("a", 1), result("x", 2)],
            "second": [result("x", 1), result("b", 2)],
        }
    )

    first = RetrievalEvaluator(retriever).evaluate(dataset)
    second = RetrievalEvaluator(retriever).evaluate(dataset)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [item.question for item in first.results] == ["first", "second"]


def test_comparison_records_recoveries_and_regressions() -> None:
    dataset = RetrievalEvalDataset(
        name="comparison",
        examples=[example("recovered", ["a"]), example("regressed", ["b"])],
    )
    dense = FakeRetriever(
        {
            "recovered": [result("x", 1)],
            "regressed": [result("b", 1)],
        }
    )
    lexical = FakeRetriever(
        {
            "recovered": [result("a", 1)],
            "regressed": [result("x", 1)],
        }
    )
    hybrid = FakeRetriever(
        {
            "recovered": [result("a", 1)],
            "regressed": [result("x", 1)],
        }
    )

    report = RetrievalComparisonEvaluator(dense, lexical, hybrid).evaluate(dataset)

    assert report.recovered_dense_failures == ["recovered"]
    assert report.dense_success_regressions == ["regressed"]
    assert report.per_question[0].dense_first_relevant_rank is None
    assert report.per_question[0].hybrid_first_relevant_rank == 1
