"""Backward-compatible imports for the original offline reranking experiment."""

from brd_knowledge.retrieval.reranking import (
    MAX_SEQUENCE_LENGTH,
    MODEL_NAME,
    MODEL_REVISION,
    PairScorer,
    RerankedChunk,
    TransformersCrossEncoderScorer,
    rerank_chunks,
)

__all__ = [
    "MAX_SEQUENCE_LENGTH",
    "MODEL_NAME",
    "MODEL_REVISION",
    "PairScorer",
    "RerankedChunk",
    "TransformersCrossEncoderScorer",
    "rerank_chunks",
]
