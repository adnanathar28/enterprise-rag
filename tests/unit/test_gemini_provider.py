from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
from pydantic import SecretStr
from tests.unit.test_grounded_answer_generation import empty_context, request

from brd_knowledge.core.exceptions import (
    GenerationBlockedError,
    GenerationProviderError,
    IncompleteGenerationError,
    InvalidCitationError,
    MalformedGenerationResponse,
    PromptBudgetExceeded,
)
from brd_knowledge.generation import GroundedAnswerService
from brd_knowledge.llm.base import LLMGenerationRequest
from brd_knowledge.llm.gemini import GeminiConfiguration, GeminiLLMProvider
from brd_knowledge.schemas.generation import ModelAnswerPayload


class GeminiTransport:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.count = 100
        self.status = 200
        self.failure: Exception | None = None
        self.body: dict[str, object] = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "answer_text": "Audit records are retained [E1].",
                                        "cited_evidence_ids": ["E1"],
                                        "insufficient_evidence": False,
                                    }
                                )
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 101,
                "candidatesTokenCount": 20,
                "thoughtsTokenCount": 3,
            },
            "modelVersion": "returned-version",
            "responseId": "response-123",
        }

    def handle(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if self.failure:
            raise self.failure
        if self.status != 200:
            return httpx.Response(
                self.status,
                json={
                    "error": {
                        "code": self.status,
                        "message": "synthetic failure",
                        "status": "UNAVAILABLE",
                    }
                },
            )
        if req.url.path.endswith(":countTokens"):
            return httpx.Response(200, json={"totalTokens": self.count})
        return httpx.Response(200, json=self.body)


@pytest.fixture
def gemini() -> Iterator[tuple[GeminiLLMProvider, GeminiTransport]]:
    transport = GeminiTransport()
    client = httpx.Client(transport=httpx.MockTransport(transport.handle))
    provider = GeminiLLMProvider(SecretStr("fake-test-key"), http_client=client)
    yield provider, transport
    provider.close()


def test_service_and_wire_requests(gemini: tuple[GeminiLLMProvider, GeminiTransport]) -> None:
    provider, transport = gemini
    result = GroundedAnswerService(provider).generate(request())
    assert result.citations[0].chunk_id == "chunk-1"
    assert result.citations[0].page_start == 4
    assert result.cited_evidence_ids == ["E1"]
    assert result.metadata.prompt_tokens == 101
    assert result.metadata.output_tokens == 20
    assert result.metadata.thinking_tokens == 3
    assert result.metadata.model_version == "returned-version"
    assert result.metadata.provider_request_id == "response-123"
    assert len(transport.requests) == 2
    count, generation = transport.requests
    assert count.url.path.endswith("/gemini-3.1-flash-lite:countTokens")
    assert generation.url.path.endswith("/gemini-3.1-flash-lite:generateContent")
    assert json.loads(count.content)["contents"] == []
    counted = json.loads(count.content)["generateContentRequest"]
    generated = json.loads(generation.content)
    assert counted["contents"] == generated["contents"]
    assert counted["systemInstruction"] == generated["systemInstruction"]
    assert counted["generationConfig"] == generated["generationConfig"]
    config = generated["generationConfig"]
    assert config["responseJsonSchema"] == ModelAnswerPayload.model_json_schema()
    assert config["responseMimeType"] == "application/json"
    assert config["temperature"] == 1.0
    assert config["candidateCount"] == 1
    assert config["maxOutputTokens"] == 256
    assert config["thinkingConfig"]["thinking_level"].lower() == "minimal"
    assert "tools" not in generated
    assert generation.extensions["timeout"]["read"] == 60.0


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_api_failures_are_not_retried(
    gemini: tuple[GeminiLLMProvider, GeminiTransport],
    status: int,
) -> None:
    provider, transport = gemini
    transport.status = status
    with pytest.raises(GenerationProviderError):
        GroundedAnswerService(provider).generate(request())
    assert len(transport.requests) == 1


def test_timeout_is_typed(gemini: tuple[GeminiLLMProvider, GeminiTransport]) -> None:
    provider, transport = gemini
    transport.failure = httpx.ReadTimeout("synthetic timeout")
    with pytest.raises(GenerationProviderError):
        GroundedAnswerService(provider).generate(request())
    assert len(transport.requests) == 1


@pytest.mark.parametrize("text", ["not json", "[]", "{}", '{"answer_text":"missing fields"}'])
def test_malformed_response(
    gemini: tuple[GeminiLLMProvider, GeminiTransport],
    text: str,
) -> None:
    provider, transport = gemini
    transport.body = {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]
    }
    with pytest.raises(MalformedGenerationResponse):
        GroundedAnswerService(provider).generate(request())


@pytest.mark.parametrize(
    "reason,error",
    [
        ("MAX_TOKENS", IncompleteGenerationError),
        ("SAFETY", GenerationBlockedError),
        ("OTHER", IncompleteGenerationError),
    ],
)
def test_abnormal_finish(
    gemini: tuple[GeminiLLMProvider, GeminiTransport],
    reason: str,
    *,
    error: type[Exception],
) -> None:
    provider, transport = gemini
    transport.body = {"candidates": [{"finishReason": reason}]}
    with pytest.raises(error):
        GroundedAnswerService(provider).generate(request())


def test_prompt_block(gemini: tuple[GeminiLLMProvider, GeminiTransport]) -> None:
    provider, transport = gemini
    transport.body = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(GenerationBlockedError):
        GroundedAnswerService(provider).generate(request())


def test_citations_still_validated(gemini: tuple[GeminiLLMProvider, GeminiTransport]) -> None:
    provider, transport = gemini
    transport.body = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "answer_text": "Invented [E99].",
                                    "cited_evidence_ids": ["E99"],
                                    "insufficient_evidence": False,
                                }
                            )
                        }
                    ]
                },
            }
        ]
    }
    with pytest.raises(InvalidCitationError):
        GroundedAnswerService(provider).generate(request())


def test_empty_context_no_calls(gemini: tuple[GeminiLLMProvider, GeminiTransport]) -> None:
    provider, transport = gemini
    assert GroundedAnswerService(provider).generate(request(empty_context())).insufficient_evidence
    assert transport.requests == []


def test_overflow_blocks_generation(gemini: tuple[GeminiLLMProvider, GeminiTransport]) -> None:
    provider, transport = gemini
    transport.count = 1_048_576
    with pytest.raises(PromptBudgetExceeded):
        GroundedAnswerService(provider).generate(request())
    assert len(transport.requests) == 1
    assert transport.requests[0].url.path.endswith(":countTokens")


def test_output_limit_blocks_all_calls(gemini: tuple[GeminiLLMProvider, GeminiTransport]) -> None:
    provider, transport = gemini
    req = request().model_copy(update={"max_output_tokens": 65_537})
    with pytest.raises(PromptBudgetExceeded):
        GroundedAnswerService(provider).generate(req)
    assert transport.requests == []


def test_generation_failure_is_not_retried(
    gemini: tuple[GeminiLLMProvider, GeminiTransport],
) -> None:
    provider, transport = gemini
    transport.status = 503
    req = LLMGenerationRequest(
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=256,
        response_schema=ModelAnswerPayload.model_json_schema(),
    )
    with pytest.raises(GenerationProviderError):
        provider.generate_structured(req)
    assert len(transport.requests) == 1


def test_configuration_and_secret_handling() -> None:
    with pytest.raises(GenerationProviderError):
        GeminiLLMProvider(SecretStr(" "))
    assert "secret-value" not in repr(SecretStr("secret-value"))
    assert GeminiConfiguration().attempts == 1


@pytest.mark.parametrize("count", [None, -1])
def test_invalid_token_count_fails_closed(gemini, count) -> None:
    provider, transport = gemini
    transport.count = count
    with pytest.raises(GenerationProviderError):
        GroundedAnswerService(provider).generate(request())
    assert len(transport.requests) == 1


def test_missing_candidate_is_malformed(gemini) -> None:
    provider, transport = gemini
    transport.body = {}
    with pytest.raises(MalformedGenerationResponse):
        GroundedAnswerService(provider).generate(request())


def test_absent_usage_is_allowed(gemini) -> None:
    provider, transport = gemini
    del transport.body["usageMetadata"]
    result = GroundedAnswerService(provider).generate(request())
    assert result.metadata.prompt_tokens == 100
    assert result.metadata.output_tokens is None
    assert result.metadata.thinking_tokens is None
