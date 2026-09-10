from __future__ import annotations

import pytest

from brd_knowledge.retrieval.experimental_reranking import rerank_chunks
from brd_knowledge.schemas.retrieval import RetrievedChunk


def candidate(chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        cosine_distance=rank / 10,
        similarity=1 - rank / 10,
        chunk_id=chunk_id,
        document_id="doc-1",
        section_path=["Section"],
        content_type="prose",
        text=f"Text for {chunk_id}",
        page_start=1,
        page_end=1,
    )


class FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return self.scores


def test_reranks_candidates_by_cross_encoder_score_and_preserves_dense_rank() -> None:
    candidates = [candidate("a", 1), candidate("b", 2), candidate("c", 3)]
    scorer = FakeScorer([0.1, 1.5, 0.8])

    results = rerank_chunks("question", candidates, scorer, top_k=2)

    assert [item.chunk.chunk_id for item in results] == ["b", "c"]
    assert [item.rank for item in results] == [1, 2]
    assert [item.dense_rank for item in results] == [2, 3]
    assert scorer.calls == [("question", ["Text for a", "Text for b", "Text for c"])]


def test_equal_scores_preserve_dense_order_before_chunk_id() -> None:
    candidates = [candidate("b", 1), candidate("a", 2)]

    results = rerank_chunks("question", candidates, FakeScorer([0.5, 0.5]), top_k=2)

    assert [item.chunk.chunk_id for item in results] == ["b", "a"]


def test_rejects_wrong_score_count() -> None:
    with pytest.raises(ValueError, match="1 scores for 2 candidates"):
        rerank_chunks(
            "question",
            [candidate("a", 1), candidate("b", 2)],
            FakeScorer([0.5]),
        )


@pytest.mark.parametrize(("query", "top_k"), [("", 5), ("question", 0)])
def test_rejects_invalid_requests(query: str, top_k: int) -> None:
    with pytest.raises(ValueError):
        rerank_chunks(query, [candidate("a", 1)], FakeScorer([0.5]), top_k=top_k)
