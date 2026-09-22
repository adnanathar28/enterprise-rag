from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from pydantic import SecretStr
from tests.unit.test_gemini_provider import GeminiTransport
from tests.unit.test_grounded_answer_generation import FakeProvider
from tests.unit.test_question_answering_service import retrieved_chunk

from brd_knowledge.core.config import Settings
from brd_knowledge.core.exceptions import InvalidCitationError, MalformedGenerationResponse
from brd_knowledge.llm.gemini import GeminiLLMProvider
from brd_knowledge.schemas.query import DocumentQuestionRequest
from brd_knowledge.services.question_answering_service import QuestionAnsweringService


def diagnostic_record(caplog: pytest.LogCaptureFixture) -> dict[str, Any]:
    messages = [
        record.message.removeprefix("grounded_generation_failure ")
        for record in caplog.records
        if record.message.startswith("grounded_generation_failure ")
    ]
    assert len(messages) == 1
    result: dict[str, Any] = json.loads(messages[0])
    return result


def question_service(
    provider: object, *, app_env: str, include_payload: bool
) -> QuestionAnsweringService:
    if isinstance(provider, FakeProvider):
        provider.close = Mock()
    retriever = Mock()
    retriever.search.return_value = [retrieved_chunk()]
    return QuestionAnsweringService(
        retriever,
        Settings(
            _env_file=None,
            APP_ENV=app_env,
            GENERATION_DIAGNOSTIC_PAYLOADS=include_payload,
            GEMINI_API_KEY="secret-test-key",
        ),
        provider_factory=lambda *_args, **_kwargs: provider,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "generated,expected_error,expected_stage",
    [
        ("not json", MalformedGenerationResponse, "json_parsing"),
        (
            json.dumps(
                {
                    "answer_text": "Invented [E99].",
                    "cited_evidence_ids": ["E99"],
                    "insufficient_evidence": False,
                }
            ),
            InvalidCitationError,
            "evidence_id_validation",
        ),
    ],
)
def test_gemini_failure_stages_and_opt_in_raw_payload(
    caplog: pytest.LogCaptureFixture,
    generated: str,
    expected_error: type[Exception],
    expected_stage: str,
) -> None:
    transport = GeminiTransport()
    transport.body["candidates"][0]["content"]["parts"][0]["text"] = generated  # type: ignore[index]
    client = httpx.Client(transport=httpx.MockTransport(transport.handle))
    provider = GeminiLLMProvider(SecretStr("secret-test-key"), http_client=client)
    service = question_service(provider, app_env="evaluation", include_payload=True)

    with caplog.at_level(logging.WARNING), pytest.raises(expected_error):
        service.answer("doc-1", DocumentQuestionRequest(question="What must be retained?"))

    record = diagnostic_record(caplog)
    assert record["stage"] == expected_stage
    assert record["exception_class"] == expected_error.__name__
    assert record["exception_message"]
    assert record["document_id"] == "doc-1"
    assert len(record["request_id"]) == 32
    assert record["model"] == "gemini-3.1-flash-lite"
    assert record["model_version"] == "returned-version"
    assert record["finish_reason"] == "STOP"
    assert record["provider_request_id"] == "response-123"
    assert record["raw_generated_text"] == generated
    assert ("parsed_payload" in record) == (expected_stage == "evidence_id_validation")
    assert "secret-test-key" not in caplog.text
    assert len(transport.requests) == 2


def test_schema_and_citation_substages_are_distinct(caplog: pytest.LogCaptureFixture) -> None:
    cases = [
        (
            {"answer_text": "Missing fields"},
            MalformedGenerationResponse,
            "model_answer_payload_validation",
        ),
        (
            {
                "answer_text": "Malformed [E1; E2].",
                "cited_evidence_ids": ["E1", "E2"],
                "insufficient_evidence": False,
            },
            InvalidCitationError,
            "inline_citation_parsing",
        ),
        (
            {
                "answer_text": "Claim [E1].",
                "cited_evidence_ids": ["E2"],
                "insufficient_evidence": False,
            },
            InvalidCitationError,
            "evidence_id_validation",
        ),
        (
            {
                "answer_text": "Claim [E1].",
                "cited_evidence_ids": [],
                "insufficient_evidence": False,
            },
            InvalidCitationError,
            "cited_evidence_ids_consistency",
        ),
    ]
    for payload, error, stage in cases:
        caplog.clear()
        service = question_service(
            FakeProvider(payload), app_env="evaluation", include_payload=True
        )
        with caplog.at_level(logging.WARNING), pytest.raises(error):
            service.answer("doc-1", DocumentQuestionRequest(question="What must be retained?"))
        record = diagnostic_record(caplog)
        assert record["stage"] == stage
        assert record["parsed_payload"] == payload


def test_missing_gemini_candidate_is_provider_extraction_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = GeminiTransport()
    transport.body["candidates"] = []
    client = httpx.Client(transport=httpx.MockTransport(transport.handle))
    provider = GeminiLLMProvider(SecretStr("secret-test-key"), http_client=client)
    service = question_service(provider, app_env="evaluation", include_payload=True)

    with caplog.at_level(logging.WARNING), pytest.raises(MalformedGenerationResponse):
        service.answer("doc-1", DocumentQuestionRequest(question="What must be retained?"))

    record = diagnostic_record(caplog)
    assert record["stage"] == "provider_response_extraction"
    assert record["model_version"] == "returned-version"
    assert record["finish_reason"] is None
    assert record["raw_generated_text"] is None
    assert "secret-test-key" not in caplog.text


def test_opt_in_payload_redacts_configured_gemini_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "answer_text": "secret-test-key [E99]",
        "cited_evidence_ids": ["E99"],
        "insufficient_evidence": False,
    }
    service = question_service(FakeProvider(payload), app_env="evaluation", include_payload=True)
    with caplog.at_level(logging.WARNING), pytest.raises(InvalidCitationError):
        service.answer("doc-1", DocumentQuestionRequest(question="What must be retained?"))
    record = diagnostic_record(caplog)
    assert record["parsed_payload"]["answer_text"] == "[redacted credential] [E99]"
    assert "secret-test-key" not in caplog.text


@pytest.mark.parametrize("app_env,include_payload", [("production", True), ("evaluation", False)])
def test_normal_logs_omit_generated_contents_and_credentials(
    caplog: pytest.LogCaptureFixture, app_env: str, include_payload: bool
) -> None:
    payload = {
        "answer_text": "Private document claim [E99].",
        "cited_evidence_ids": ["E99"],
        "insufficient_evidence": False,
    }
    service = question_service(
        FakeProvider(payload), app_env=app_env, include_payload=include_payload
    )
    with caplog.at_level(logging.WARNING), pytest.raises(InvalidCitationError):
        service.answer("doc-1", DocumentQuestionRequest(question="What must be retained?"))
    record = diagnostic_record(caplog)
    assert record["stage"] == "evidence_id_validation"
    assert record["exception_message"] == "Generated answer cites unknown evidence IDs: [redacted]"
    assert "raw_generated_text" not in record
    assert "parsed_payload" not in record
    assert "Private document claim" not in caplog.text
    assert "secret-test-key" not in caplog.text


def test_success_has_no_failure_diagnostic(caplog: pytest.LogCaptureFixture) -> None:
    service = question_service(FakeProvider(), app_env="evaluation", include_payload=True)
    with caplog.at_level(logging.WARNING):
        response = service.answer(
            "doc-1", DocumentQuestionRequest(question="What must be retained?")
        )
    assert response.answer.answer_text == "Audit records are retained [E1]."
    assert response.answer.citations[0].chunk_id == "chunk-1"
    assert not any("grounded_generation_failure" in record.message for record in caplog.records)
