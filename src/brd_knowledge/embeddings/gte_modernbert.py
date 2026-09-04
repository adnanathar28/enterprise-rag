from __future__ import annotations

import math
from importlib import import_module
from typing import Any, cast

from brd_knowledge.embeddings.base import EmbeddingConfiguration

MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"
MODEL_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
EMBEDDING_DIMENSION = 768
MAX_SEQUENCE_LENGTH = 8192


class EmbeddingTokenLimitError(ValueError):
    pass


class EmbeddingOutputError(ValueError):
    pass


class GteModernBertEmbeddingProvider:
    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        model_revision: str = MODEL_REVISION,
        dimension: int = EMBEDDING_DIMENSION,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        batch_size: int = 16,
        device: str = "cpu",
        preprocessing_version: str = "1",
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be greater than zero.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        self._configuration = EmbeddingConfiguration(
            provider="huggingface-transformers",
            model_name=model_name,
            model_revision=model_revision,
            dimension=dimension,
            max_sequence_length=max_sequence_length,
            pooling="cls",
            normalize_embeddings=True,
            output_dtype="float32",
            similarity_metric="cosine",
            trust_remote_code=False,
            query_prefix="",
            document_prefix="",
            preprocessing_version=preprocessing_version,
        )
        self._batch_size = batch_size
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module

    @property
    def configuration(self) -> EmbeddingConfiguration:
        return self._configuration

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._tokenizer is not None
        for index, text in enumerate(texts):
            token_count = len(
                self._tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            if token_count > self.configuration.max_sequence_length:
                raise EmbeddingTokenLimitError(
                    f"Document text at index {index} has {token_count} tokens; "
                    f"maximum is {self.configuration.max_sequence_length}."
                )

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._encode_batch(texts[start : start + self._batch_size]))
        self._validate_vectors(vectors, expected_count=len(texts))
        return vectors

    def _ensure_loaded(self) -> None:
        if self._torch is None:
            self._torch = import_module("torch")
        if self._tokenizer is not None and self._model is not None:
            return
        transformers = import_module("transformers")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.configuration.model_name,
            revision=self.configuration.model_revision,
            trust_remote_code=False,
        )
        self._model = transformers.AutoModel.from_pretrained(
            self.configuration.model_name,
            revision=self.configuration.model_revision,
            trust_remote_code=False,
        )
        self._model.to(self._device)
        self._model.eval()

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        encoded = self._tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded = {name: value.to(self._device) for name, value in encoded.items()}
        with self._torch.no_grad():
            outputs = self._model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0]
            normalized = self._torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
        vectors = normalized.detach().to(dtype=self._torch.float32).cpu().tolist()
        return cast(list[list[float]], vectors)

    def _validate_vectors(self, vectors: list[list[float]], expected_count: int) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingOutputError(
                f"Provider returned {len(vectors)} vectors for {expected_count} texts."
            )
        for index, vector in enumerate(vectors):
            if len(vector) != self.configuration.dimension:
                raise EmbeddingOutputError(
                    f"Vector {index} has dimension {len(vector)}; "
                    f"expected {self.configuration.dimension}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingOutputError(f"Vector {index} contains a non-finite value.")
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise EmbeddingOutputError(
                    f"Vector {index} is not normalized; L2 norm is {norm:.6f}."
                )
