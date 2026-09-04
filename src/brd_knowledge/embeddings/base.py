from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingConfiguration:
    provider: str
    model_name: str
    model_revision: str
    dimension: int
    max_sequence_length: int
    pooling: str
    normalize_embeddings: bool
    output_dtype: str
    similarity_metric: str
    trust_remote_code: bool
    query_prefix: str
    document_prefix: str
    preprocessing_version: str

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EmbeddingProvider(Protocol):
    @property
    def configuration(self) -> EmbeddingConfiguration:
        """Return the complete identity of the active embedding space."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts in input order."""
