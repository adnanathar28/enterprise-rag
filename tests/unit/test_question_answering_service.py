from unittest.mock import Mock

import pytest
from tests.unit.test_grounded_answer_generation import FakeProvider

from brd_knowledge.core.config import Settings
from brd_knowledge.core.exceptions import InvalidCitationError
from brd_knowledge.schemas.query import DocumentQuestionRequest
from brd_knowledge.schemas.retrieval import RetrievedChunk
from brd_knowledge.services.question_answering_service import QuestionAnsweringService


def retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        cosine_distance=0.1,
        similarity=0.9,
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="Audit",
        section_path=["Controls", "Audit"],
        content_type="prose",
        text="Audit records must be retained.",
        page_start=7,
        page_end=7,
        source_block_ids=["block-1"],
    )


def test_orchestrates_document_scoped_dense_retrieval_and_grounded_generation() -> None:
    retriever = Mock()
    retriever.search.return_value = [retrieved_chunk()]
    provider = FakeProvider()
    provider.close = Mock()
    factory = Mock(return_value=provider)
    settings = Settings(_env_file=None)
    service = QuestionAnsweringService(retriever, settings, provider_factory=factory)

    result = service.answer(
        "doc-1",
        DocumentQuestionRequest(
            question="What is retained?",
            provider="local_qwen",
            model="qwen3:8b",
        ),
        top_k=3,
        max_context_characters=4_000,
        max_output_tokens=256,
    )

    retriever.search.assert_called_once_with(
        "What is retained?",
        top_k=3,
        document_id="doc-1",
    )
    factory.assert_called_once_with(
        settings,
        provider="local_qwen",
        model="qwen3:8b",
    )
    assert result.context.evidence[0].evidence_id == "E1"
    assert result.answer.citations[0].chunk_id == "chunk-1"
    assert result.answer.citations[0].page_start == 7
    assert result.retrieved_chunks[0].similarity == 0.9
    assert provider.requests[0].max_output_tokens == 256
    provider.close.assert_called_once_with()


def test_preserves_application_side_citation_validation_and_closes_provider() -> None:
    retriever = Mock()
    retriever.search.return_value = [retrieved_chunk()]
    provider = FakeProvider(
        {
            "answer_text": "Invented evidence [E99].",
            "cited_evidence_ids": ["E99"],
            "insufficient_evidence": False,
        }
    )
    provider.close = Mock()
    service = QuestionAnsweringService(
        retriever,
        Settings(_env_file=None),
        provider_factory=Mock(return_value=provider),
    )

    with pytest.raises(InvalidCitationError):
        service.answer("doc-1", DocumentQuestionRequest(question="What is retained?"))

    provider.close.assert_called_once_with()
