from __future__ import annotations

import pytest
from pydantic import ValidationError

from brd_knowledge.context import ContextBuilder
from brd_knowledge.core.exceptions import (
    GenerationProviderError,
    InvalidCitationError,
    MalformedGenerationResponse,
    PromptBudgetExceeded,
)
from brd_knowledge.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from brd_knowledge.generation.service import GroundedAnswerService
from brd_knowledge.llm.base import (
    LLMConfiguration,
    LLMGenerationRequest,
    LLMGenerationResponse,
    LLMProvider,
)
from brd_knowledge.schemas.context import ConstructedContext, ContextBuildRequest
from brd_knowledge.schemas.generation import GroundedAnswerRequest, ModelAnswerPayload
from brd_knowledge.schemas.retrieval import RetrievedChunk


class FakeProvider(LLMProvider):
    def __init__(
        self,
        payload: dict[str, object] | None = None,
        *,
        token_count: int = 100,
        failure: Exception | None = None,
    ) -> None:
        self.payload = payload or {
            "answer_text": "Audit records are retained [E1].",
            "cited_evidence_ids": ["E1"],
            "insufficient_evidence": False,
        }
        self.token_count = token_count
        self.failure = failure
        self.requests: list[LLMGenerationRequest] = []
        self.count_calls: list[tuple[str, str]] = []

    @property
    def configuration(self) -> LLMConfiguration:
        return LLMConfiguration(
            provider="fake",
            model="fake-model",
            revision="revision-1",
            context_window_tokens=2_000,
            safety_margin_tokens=100,
        )

    def count_tokens(self, system_prompt: str, user_prompt: str) -> int:
        self.count_calls.append((system_prompt, user_prompt))
        return self.token_count

    def generate_structured(self, request: LLMGenerationRequest) -> LLMGenerationResponse:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return LLMGenerationResponse(
            payload=self.payload,
            prompt_tokens=self.token_count,
            output_tokens=12,
            request_id="request-1",
        )


def constructed_context() -> ConstructedContext:
    retrieved = RetrievedChunk(
        rank=1,
        cosine_distance=0.1,
        similarity=0.9,
        chunk_id="chunk-1",
        document_id="doc-1",
        section_id="section-1",
        section_title="Audit",
        section_path=["Controls", "Audit"],
        content_type="prose",
        text="The system shall retain audit records.",
        page_start=4,
        page_end=5,
        source_block_ids=["block-1"],
        source_table_ids=[],
        provenance=[
            {
                "document_id": "doc-1",
                "page_number": 4,
                "block_id": "block-1",
                "reading_order_index": 8,
            }
        ],
        quality_notes=["review source"],
    )
    return ContextBuilder().build(
        ContextBuildRequest(retrieved_chunks=[retrieved], max_characters=5_000)
    )


def empty_context() -> ConstructedContext:
    return ContextBuilder().build(
        ContextBuildRequest(retrieved_chunks=[], max_characters=5_000)
    )


def request(context: ConstructedContext | None = None) -> GroundedAnswerRequest:
    return GroundedAnswerRequest(
        question="What audit records are required?",
        context=context or constructed_context(),
        max_output_tokens=256,
    )


def test_prompt_is_stable_and_delimits_untrusted_evidence() -> None:
    context = constructed_context()

    prompt = build_user_prompt("What is required?", context)

    assert "only the supplied evidence" in SYSTEM_PROMPT
    assert "untrusted source material" in SYSTEM_PROMPT
    assert prompt == (
        "QUESTION\n\nWhat is required?\n\nEVIDENCE\n\n<evidence>\n"
        f"{context.rendered_text}\n</evidence>"
    )


def test_generates_valid_answer_and_resolves_authoritative_citation() -> None:
    provider = FakeProvider()

    result = GroundedAnswerService(provider).generate(request())

    assert result.answer_text == "Audit records are retained [E1]."
    assert result.cited_evidence_ids == ["E1"]
    assert result.citations[0].chunk_id == "chunk-1"
    assert result.citations[0].page_start == 4
    assert result.citations[0].source_block_ids == ["block-1"]
    assert result.citations[0].provenance[0].reading_order_index == 8
    assert result.metadata.model == "fake-model"
    assert result.metadata.prompt_tokens == 100
    assert result.metadata.output_tokens == 12
    assert len(provider.requests) == 1
    assert provider.requests[0].response_schema == ModelAnswerPayload.model_json_schema()


def test_duplicate_citations_are_deduplicated_in_first_use_order() -> None:
    provider = FakeProvider(
        {
            "answer_text": "Required [E1], with confirmation [E1].",
            "cited_evidence_ids": ["E1", "E1"],
            "insufficient_evidence": False,
        }
    )

    result = GroundedAnswerService(provider).generate(request())

    assert result.cited_evidence_ids == ["E1"]
    assert len(result.citations) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "answer_text": "Unsupported [E99].",
            "cited_evidence_ids": ["E99"],
            "insufficient_evidence": False,
        },
        {
            "answer_text": "Supported [E1], but invented [SOURCE1].",
            "cited_evidence_ids": ["E1"],
            "insufficient_evidence": False,
        },
        {
            "answer_text": "Supported [E1].",
            "cited_evidence_ids": [],
            "insufficient_evidence": False,
        },
        {
            "answer_text": "An answer without a citation.",
            "cited_evidence_ids": [],
            "insufficient_evidence": False,
        },
    ],
)
def test_rejects_unknown_mismatched_or_missing_citations(
    payload: dict[str, object],
) -> None:
    with pytest.raises(InvalidCitationError):
        GroundedAnswerService(FakeProvider(payload)).generate(request())


def test_accepts_citation_free_insufficient_evidence_response() -> None:
    provider = FakeProvider(
        {
            "answer_text": "The supplied evidence does not specify the retention period.",
            "cited_evidence_ids": [],
            "insufficient_evidence": True,
        }
    )

    result = GroundedAnswerService(provider).generate(request())

    assert result.insufficient_evidence is True
    assert result.citations == []


def test_empty_context_returns_deterministic_insufficient_answer_without_provider_call() -> None:
    provider = FakeProvider()

    result = GroundedAnswerService(provider).generate(request(empty_context()))

    assert result.insufficient_evidence is True
    assert result.cited_evidence_ids == []
    assert provider.count_calls == []
    assert provider.requests == []


def test_rejects_malformed_provider_payload() -> None:
    provider = FakeProvider({"answer_text": "Missing required fields"})

    with pytest.raises(MalformedGenerationResponse):
        GroundedAnswerService(provider).generate(request())


def test_rejects_model_supplied_source_metadata() -> None:
    provider = FakeProvider(
        {
            "answer_text": "Supported [E1].",
            "cited_evidence_ids": ["E1"],
            "insufficient_evidence": False,
            "page_number": 999,
        }
    )

    with pytest.raises(MalformedGenerationResponse):
        GroundedAnswerService(provider).generate(request())


def test_rejects_prompt_overflow_before_generation() -> None:
    provider = FakeProvider(token_count=1_700)

    with pytest.raises(PromptBudgetExceeded, match="2056 tokens"):
        GroundedAnswerService(provider).generate(request())

    assert provider.requests == []


def test_wraps_provider_failure() -> None:
    provider = FakeProvider(failure=RuntimeError("offline"))

    with pytest.raises(GenerationProviderError):
        GroundedAnswerService(provider).generate(request())


def test_rejects_blank_question_before_generation() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswerRequest(question="   ", context=constructed_context())
