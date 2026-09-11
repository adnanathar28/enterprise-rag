from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

from brd_knowledge.core.exceptions import RetrievalRerankingError
from brd_knowledge.schemas.retrieval import RetrievedChunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MAX_SEQUENCE_LENGTH = 512


class PairScorer(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    rank: int
    reranker_score: float
    dense_rank: int
    chunk: RetrievedChunk


class TransformersCrossEncoderScorer:
    """Score query/passage pairs with a pinned local cross-encoder."""

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        model_revision: str = MODEL_REVISION,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        batch_size: int = 16,
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be greater than zero.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        self.model_name = model_name
        self.model_revision = model_revision
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.device = device
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if not passages:
            return []
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            encoded = self._tokenizer(
                [query] * len(batch),
                list(batch),
                add_special_tokens=True,
                padding=True,
                truncation="only_second",
                max_length=self.max_sequence_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with self._torch.no_grad():
                logits = self._model(**encoded).logits.reshape(-1)
            scores.extend(
                cast(list[float], logits.detach().to(dtype=self._torch.float32).cpu().tolist())
            )
        return scores

    def load(self) -> None:
        """Load and validate the pinned model before the first scoring call."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._torch is None:
            self._torch = import_module("torch")
        if self._tokenizer is not None and self._model is not None:
            return
        transformers = import_module("transformers")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            trust_remote_code=False,
        )
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            trust_remote_code=False,
        )
        self._model.to(self.device)
        self._model.eval()


class CrossEncoderRerankingRetriever:
    """Rerank a fixed dense candidate pool and return standard retrieval results."""

    def __init__(
        self,
        dense_retriever: Retriever,
        scorer: PairScorer,
        *,
        candidate_k: int = 40,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")
        self._dense_retriever = dense_retriever
        self._scorer = scorer
        self._candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        candidates = self._dense_retriever.search(
            query,
            top_k=max(self._candidate_k, top_k),
            document_id=document_id,
        )
        try:
            reranked = rerank_chunks(query, candidates, self._scorer, top_k=top_k)
        except Exception as exc:
            raise RetrievalRerankingError("Cross-encoder reranking failed.") from exc
        return [item.chunk.model_copy(update={"rank": item.rank}) for item in reranked]


def rerank_chunks(
    query: str,
    candidates: Sequence[RetrievedChunk],
    scorer: PairScorer,
    *,
    top_k: int = 5,
) -> list[RerankedChunk]:
    if not query.strip():
        raise ValueError("Query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    scores = scorer.score(query, [candidate.text for candidate in candidates])
    if len(scores) != len(candidates):
        raise ValueError(
            f"Reranker returned {len(scores)} scores for {len(candidates)} candidates."
        )
    scored = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: (-item[1], item[0].rank, item[0].chunk_id),
    )
    return [
        RerankedChunk(
            rank=rank,
            reranker_score=float(score),
            dense_rank=chunk.rank,
            chunk=chunk,
        )
        for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
    ]
