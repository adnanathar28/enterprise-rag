from __future__ import annotations

import json
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError
from tests.unit.test_grounded_answer_generation import empty_context, request

from brd_knowledge.core.exceptions import (
    GenerationProviderError,
    IncompleteGenerationError,
    InvalidCitationError,
    MalformedGenerationResponse,
    PromptBudgetExceeded,
)
from brd_knowledge.generation import GroundedAnswerService
from brd_knowledge.llm.local_qwen import LocalQwenConfiguration, LocalQwenProvider
from brd_knowledge.schemas.generation import ModelAnswerPayload


@pytest.fixture
def local():
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "exact formatted prompt"
    tokenizer.encode.return_value = [1] * 100
    requests = []
    body = {
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
        "response": json.dumps(
            {
                "answer_text": "Retain audit records [E1].",
                "cited_evidence_ids": ["E1"],
                "insufficient_evidence": False,
            }
        ),
    }

    def handle(req):
        requests.append(req)
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "digest": "digest"}]})
        return httpx.Response(200, json=body)

    provider = LocalQwenProvider(tokenizer=tokenizer, transport=httpx.MockTransport(handle))
    yield provider, tokenizer, requests, body
    provider.close()


def test_complete_local_answer_and_exact_wire_prompt(local) -> None:
    provider, tokenizer, requests, body = local
    answer = GroundedAnswerService(provider).generate(request())
    wire = json.loads(requests[-1].content)
    assert wire["prompt"] == tokenizer.encode.call_args.args[0] == "exact formatted prompt"
    assert wire["format"] == ModelAnswerPayload.model_json_schema()
    assert wire["raw"] is True and wire["stream"] is False and wire["think"] is False
    assert wire["options"]["num_ctx"] == 8192
    assert wire["options"]["num_predict"] == 256
    assert wire["options"]["seed"] == 42
    assert tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"] is False
    assert answer.citations[0].chunk_id == "chunk-1"
    assert answer.metadata.model_version == "digest"
    assert answer.metadata.prompt_tokens == 100
    assert answer.metadata.output_tokens == 20
    assert answer.metadata.provider_request_id is None


def test_empty_context_does_not_load_tokenizer_or_call_ollama(local) -> None:
    provider, tokenizer, requests, body = local
    assert GroundedAnswerService(provider).generate(request(empty_context())).insufficient_evidence
    tokenizer.apply_chat_template.assert_not_called()
    assert requests == []


def test_overflow_prevents_all_http(local) -> None:
    provider, tokenizer, requests, body = local
    tokenizer.encode.return_value = [1] * 8000
    with pytest.raises(PromptBudgetExceeded):
        GroundedAnswerService(provider).generate(request())
    assert requests == []


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"response": "not json"}, MalformedGenerationResponse),
        ({"response": "[]"}, MalformedGenerationResponse),
        ({"response": "{}"}, MalformedGenerationResponse),
        ({"done_reason": "length"}, IncompleteGenerationError),
        ({"prompt_eval_count": 99}, GenerationProviderError),
        ({"prompt_eval_count": None}, GenerationProviderError),
        ({"eval_count": -1}, MalformedGenerationResponse),
        (
            {
                "response": json.dumps(
                    {
                        "answer_text": "Invented [E99]",
                        "cited_evidence_ids": ["E99"],
                        "insufficient_evidence": False,
                    }
                )
            },
            InvalidCitationError,
        ),
    ],
)
def test_invalid_responses_preserve_typed_failures(local, patch, error) -> None:
    provider, tokenizer, requests, body = local
    body.update(patch)
    with pytest.raises(error):
        GroundedAnswerService(provider).generate(request())


def test_transport_failure_is_not_retried() -> None:
    calls = []

    def handle(req):
        calls.append(req)
        raise httpx.ConnectError("offline")

    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "prompt"
    tokenizer.encode.return_value = [1]
    provider = LocalQwenProvider(tokenizer=tokenizer, transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(GenerationProviderError):
            GroundedAnswerService(provider).generate(request())
        assert len(calls) == 1
    finally:
        provider.close()


@pytest.mark.parametrize(
    "url", ["https://ollama.com", "http://example.com:11434", "http://user:secret@localhost:11434"]
)
def test_local_provider_rejects_remote_hosts(url) -> None:
    with pytest.raises(ValidationError):
        LocalQwenConfiguration(base_url=url)
