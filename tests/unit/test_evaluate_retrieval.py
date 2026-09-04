import importlib.util
from pathlib import Path
from types import ModuleType

from brd_knowledge.schemas.evaluation import (
    RetrievalEvaluationReport,
    RetrievalExampleResult,
    RetrievalMetrics,
    RetrievedEvidence,
)


def load_script() -> ModuleType:
    script_path = Path("scripts/evaluate_retrieval.py")
    spec = importlib.util.spec_from_file_location("evaluate_retrieval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_highlights_failures_and_records_retrieved_evidence() -> None:
    report = RetrievalEvaluationReport(
        dataset_name="sample",
        metrics=RetrievalMetrics(
            example_count=1,
            recall_at_1=0,
            recall_at_3=0,
            recall_at_5=0,
            mrr=0,
        ),
        results=[
            RetrievalExampleResult(
                question="Where is the evidence?",
                expected_document_id="doc-1",
                relevant_chunk_ids=["expected"],
                section_page_targets=[],
                retrieved=[
                    RetrievedEvidence(
                        rank=1,
                        chunk_id="wrong",
                        document_id="doc-1",
                        section_path=["Wrong section"],
                        page_start=4,
                        page_end=4,
                        similarity=0.5,
                        relevant=False,
                    )
                ],
                first_relevant_rank=None,
                hit_at_1=False,
                hit_at_3=False,
                hit_at_5=False,
            )
        ],
    )

    output = load_script().render_report(report)

    assert "Recall@5 0.000" in output
    assert "[1] FAIL first_relevant_rank=NONE" in output
    assert "expected=expected" in output
    assert "chunk=wrong pages=4-4 section=Wrong section" in output


def test_real_dataset_loads_with_twenty_labeled_examples() -> None:
    dataset = load_script().load_dataset(Path("data/evals/retrieval_eval.json"))

    assert len(dataset.examples) == 20
    assert all(example.relevant_chunk_ids for example in dataset.examples)
