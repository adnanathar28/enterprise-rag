import importlib.util
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import SecretStr
from tests.unit.test_grounded_answer_generation import FakeProvider

from brd_knowledge.core.config import Settings
from brd_knowledge.core.exceptions import GenerationProviderError
from brd_knowledge.llm import factory
from brd_knowledge.llm.gemini import GeminiConfiguration
from brd_knowledge.schemas.retrieval import RetrievedChunk


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "answer_question", Path("scripts/answer_question.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_uses_dense_retrieval_and_resolves_citations(monkeypatch, capsys) -> None:
    script = load_script()
    monkeypatch.setattr(
        "sys.argv",
        ["answer_question", "What is required?", "--document-id", "approved-doc", "--top-k", "3"],
    )
    settings = Settings(_env_file=None, GEMINI_API_KEY=SecretStr("fake-test-key"))
    monkeypatch.setattr(script, "get_settings", lambda: settings)
    embedding = Mock()
    embedding_factory = Mock(return_value=embedding)
    monkeypatch.setattr(script, "GteModernBertEmbeddingProvider", embedding_factory)
    session = MagicMock()
    monkeypatch.setattr(script, "SessionLocal", Mock(return_value=session))
    retriever = Mock()
    retriever.search.return_value = [
        RetrievedChunk(
            rank=1,
            cosine_distance=0.1,
            similarity=0.9,
            chunk_id="retained-chunk",
            document_id="approved-doc",
            content_type="prose",
            text="Audit records must be retained.",
            page_start=7,
            page_end=7,
        )
    ]
    dense_factory = Mock(return_value=retriever)
    monkeypatch.setattr(script, "PgVectorRetriever", dense_factory)
    provider = FakeProvider()
    provider.close = Mock()
    provider_factory = Mock(return_value=provider)
    monkeypatch.setattr(script, "create_llm_provider", provider_factory)

    script.main()

    dense_factory.assert_called_once_with(session.__enter__.return_value, embedding)
    retriever.search.assert_called_once_with(
        "What is required?", top_k=3, document_id="approved-doc"
    )
    assert embedding_factory.call_args.kwargs["model_revision"] == settings.embedding_model_revision
    assert provider.requests[0].max_output_tokens == 1024
    assert "Audit records must be retained." in provider.requests[0].user_prompt
    output = json.loads(capsys.readouterr().out)
    assert output["answer"]["citations"][0]["chunk_id"] == "retained-chunk"
    assert output["answer"]["citations"][0]["page_start"] == 7
    assert output["context"]["included_chunks"] == 1
    assert output["configuration"]["provider"] == "fake"
    assert "fake-test-key" not in json.dumps(output)
    provider.close.assert_called_once()


def test_cli_missing_key_fails_before_retrieval(monkeypatch) -> None:
    script = load_script()
    monkeypatch.setattr("sys.argv", ["answer_question", "Question?", "--document-id", "doc"])
    monkeypatch.setattr(
        script, "get_settings", lambda: Settings(_env_file=None, GEMINI_API_KEY=None)
    )
    embedding_factory = Mock()
    monkeypatch.setattr(script, "GteModernBertEmbeddingProvider", embedding_factory)
    with pytest.raises(GenerationProviderError):
        script.main()
    embedding_factory.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        ["Question?"],
        [" ", "--document-id", "doc"],
        ["Question?", "--document-id", "doc", "--top-k", "0"],
        ["Question?", "--document-id", "doc", "--max-characters", "-1"],
    ],
)
def test_cli_rejects_invalid_arguments(monkeypatch, args) -> None:
    monkeypatch.setattr("sys.argv", ["answer_question", *args])
    with pytest.raises(SystemExit):
        load_script().parse_args()


@pytest.mark.parametrize("selection", [None, "local_qwen"])
def test_local_factory_never_requires_or_constructs_gemini(monkeypatch, selection) -> None:
    settings = Settings(_env_file=None, LLM_PROVIDER="local_qwen", GEMINI_API_KEY=None)
    local = Mock()
    remote = Mock(side_effect=AssertionError("Gemini must not be constructed"))
    monkeypatch.setattr(factory, "LocalQwenProvider", local)
    monkeypatch.setattr(factory, "GeminiLLMProvider", remote)
    assert (
        factory.create_llm_provider(settings, provider=selection, model="qwen3:8b")
        == local.return_value
    )
    assert local.call_args.args[0].context_window_tokens == 8192
    remote.assert_not_called()


def test_gemini_override_retains_configuration(monkeypatch) -> None:
    settings = Settings(_env_file=None, LLM_PROVIDER="local_qwen", GEMINI_API_KEY="fake")
    remote = Mock()
    monkeypatch.setattr(factory, "GeminiLLMProvider", remote)
    factory.create_llm_provider(settings, provider="gemini")
    assert isinstance(remote.call_args.args[1], GeminiConfiguration)
