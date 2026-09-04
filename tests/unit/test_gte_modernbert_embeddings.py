import pytest

from brd_knowledge.embeddings.gte_modernbert import (
    EMBEDDING_DIMENSION,
    EmbeddingTokenLimitError,
    GteModernBertEmbeddingProvider,
)


class FakeTokenizer:
    def encode(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> list[int]:
        assert add_special_tokens is True
        assert truncation is False
        return list(range(int(text)))


class RecordingProvider(GteModernBertEmbeddingProvider):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            tokenizer=FakeTokenizer(),
            model=object(),
            torch_module=object(),
            **kwargs,
        )
        self.batches: list[list[str]] = []

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in texts]


def test_batches_documents_and_returns_normalized_768d_vectors() -> None:
    provider = RecordingProvider(batch_size=2)

    vectors = provider.embed_documents(["2", "3", "4"])

    assert provider.batches == [["2", "3"], ["4"]]
    assert len(vectors) == 3
    assert all(len(vector) == 768 for vector in vectors)
    assert provider.configuration.pooling == "cls"
    assert provider.configuration.normalize_embeddings is True
    assert provider.configuration.output_dtype == "float32"
    assert provider.configuration.similarity_metric == "cosine"
    assert provider.configuration.trust_remote_code is False
    assert provider.configuration.query_prefix == ""
    assert provider.configuration.document_prefix == ""


def test_rejects_over_limit_input_without_silent_truncation() -> None:
    provider = RecordingProvider(max_sequence_length=8192)

    with pytest.raises(EmbeddingTokenLimitError, match="8193 tokens"):
        provider.embed_documents(["8193"])

    assert provider.batches == []


def test_embedding_configuration_hash_changes_with_preprocessing() -> None:
    first = RecordingProvider(preprocessing_version="1")
    changed = RecordingProvider(preprocessing_version="2")

    assert first.configuration.config_hash != changed.configuration.config_hash
