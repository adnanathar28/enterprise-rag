from __future__ import annotations

from typing import Any

import pytest

from brd_knowledge.database.models.chunk_embedding import ChunkEmbedding
from brd_knowledge.embeddings.base import EmbeddingConfiguration
from brd_knowledge.schemas.chunk import Chunk
from brd_knowledge.schemas.source import SourceReference
from brd_knowledge.services.chunk_embedding_service import ChunkEmbeddingService


class FakeProvider:
    def __init__(self, *, revision: str = "revision-1", preprocessing: str = "1") -> None:
        self.configuration = EmbeddingConfiguration(
            provider="fake",
            model_name="fake-model",
            model_revision=revision,
            dimension=768,
            max_sequence_length=8192,
            pooling="cls",
            normalize_embeddings=True,
            output_dtype="float32",
            similarity_metric="cosine",
            trust_remote_code=False,
            query_prefix="",
            document_prefix="",
            preprocessing_version=preprocessing,
        )
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0] + [0.0] * 767 for _ in texts]


class FakeQuery:
    def __init__(self, rows: list[Any], *, one: Any | None = None) -> None:
        self.rows = rows
        self.one = one

    def filter(self, *_criteria: object) -> FakeQuery:
        return self

    def all(self) -> list[ChunkEmbedding]:
        return list(self.rows)

    def one_or_none(self) -> Any | None:
        return self.one


class FakeSession:
    def __init__(self) -> None:
        self.rows: list[ChunkEmbedding] = []
        self.commits = 0
        self.rollbacks = 0

    def query(self, model: type[Any]) -> FakeQuery:
        if model is not ChunkEmbedding:
            return FakeQuery([], one=object())
        return FakeQuery(self.rows)

    def add(self, row: ChunkEmbedding) -> None:
        self.rows.append(row)

    def delete(self, row: ChunkEmbedding) -> None:
        self.rows.remove(row)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def make_chunk(chunk_id: str, text: str = "Requirement text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        section_id="section-1",
        section_title="1) Scope",
        section_path=["1) Scope"],
        content_type="prose",
        text=text,
        page_start=1,
        page_end=2,
        source_block_ids=["block-1"],
        provenance=[
            SourceReference(
                document_id="doc-1",
                page_number=1,
                block_id="block-1",
                reading_order_index=4,
            )
        ],
        quality_notes=["example-note"],
    )


def sync(session: FakeSession, provider: FakeProvider, chunks: list[Chunk]) -> Any:
    return ChunkEmbeddingService(session, provider).synchronize("doc-1", chunks)  # type: ignore[arg-type]


def test_first_sync_persists_complete_traceable_metadata() -> None:
    session = FakeSession()
    provider = FakeProvider()
    chunk = make_chunk("chunk-1")

    result = sync(session, provider, [chunk])

    assert (result.embedded, result.updated, result.skipped, result.deleted) == (1, 0, 0, 0)
    row = session.rows[0]
    assert row.text == chunk.text
    assert row.section_path == ["1) Scope"]
    assert row.source_block_ids == ["block-1"]
    assert row.source_table_ids == []
    assert row.provenance[0]["reading_order_index"] == 4
    assert row.quality_notes == ["example-note"]
    assert len(row.embedding) == 768
    assert len(row.content_hash) == 64


def test_identical_second_sync_skips_embedding() -> None:
    session = FakeSession()
    first_provider = FakeProvider()
    sync(session, first_provider, [make_chunk("chunk-1")])
    second_provider = FakeProvider()

    result = sync(session, second_provider, [make_chunk("chunk-1")])

    assert (result.embedded, result.updated, result.skipped, result.deleted) == (0, 0, 1, 0)
    assert second_provider.calls == []


def test_metadata_refresh_does_not_reembed_unchanged_text() -> None:
    session = FakeSession()
    sync(session, FakeProvider(), [make_chunk("chunk-1")])
    changed_metadata = make_chunk("chunk-1")
    changed_metadata.quality_notes = ["new-quality-note"]
    provider = FakeProvider()

    result = sync(session, provider, [changed_metadata])

    assert result.skipped == 1
    assert provider.calls == []
    assert session.rows[0].quality_notes == ["new-quality-note"]


@pytest.mark.parametrize(
    ("provider", "chunk"),
    [
        (FakeProvider(revision="revision-2"), make_chunk("chunk-1")),
        (FakeProvider(preprocessing="2"), make_chunk("chunk-1")),
        (FakeProvider(), make_chunk("chunk-1", "Changed requirement")),
    ],
)
def test_model_config_or_content_change_reembeds(
    provider: FakeProvider, chunk: Chunk
) -> None:
    session = FakeSession()
    sync(session, FakeProvider(), [make_chunk("chunk-1")])

    result = sync(session, provider, [chunk])

    assert (result.embedded, result.updated, result.skipped) == (0, 1, 0)
    assert provider.calls == [[chunk.text]]


def test_stale_vectors_are_deleted() -> None:
    session = FakeSession()
    sync(session, FakeProvider(), [make_chunk("chunk-1"), make_chunk("chunk-2")])

    result = sync(session, FakeProvider(), [make_chunk("chunk-2")])

    assert result.deleted == 1
    assert [row.chunk_id for row in session.rows] == ["chunk-2"]


def test_duplicate_chunk_ids_are_rejected_before_transaction() -> None:
    session = FakeSession()

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        sync(session, FakeProvider(), [make_chunk("chunk-1"), make_chunk("chunk-1")])

    assert session.commits == 0


def test_provider_failure_rolls_back_transaction() -> None:
    class FailingProvider(FakeProvider):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding failed")

    session = FakeSession()

    with pytest.raises(RuntimeError, match="embedding failed"):
        sync(session, FailingProvider(), [make_chunk("chunk-1")])

    assert session.commits == 0
    assert session.rollbacks == 1
