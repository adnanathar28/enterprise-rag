import importlib.util
from pathlib import Path
from types import ModuleType

from brd_knowledge.schemas.evaluation import (
    QueryStrategyComparison,
    RetrievalComparisonReport,
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
    strategy_report = RetrievalEvaluationReport(
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
                        score=0.5,
                        strategy="dense",
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
    report = RetrievalComparisonReport(
        dense=strategy_report,
        lexical=strategy_report.model_copy(deep=True),
        hybrid=strategy_report.model_copy(deep=True),
        per_question=[
            QueryStrategyComparison(
                question="Where is the evidence?",
                dense_first_relevant_rank=None,
                lexical_first_relevant_rank=None,
                hybrid_first_relevant_rank=None,
            )
        ],
        recovered_dense_failures=[],
        dense_success_regressions=[],
    )

    output = load_script().render_report(report)

    assert "dense             0.000     0.000     0.000  0.000" in output
    assert "[1] dense=NONE lexical=NONE hybrid=NONE" in output
    assert "RECOVERED DENSE TOP-5 FAILURES\n- NONE" in output
    assert "DENSE TOP-5 SUCCESSES REGRESSED BY HYBRID\n- NONE" in output


def test_real_dataset_loads_with_twenty_labeled_examples() -> None:
    dataset = load_script().load_dataset(Path("data/evals/retrieval_eval.json"))

    assert len(dataset.examples) == 20
    assert all(example.relevant_chunk_ids for example in dataset.examples)
