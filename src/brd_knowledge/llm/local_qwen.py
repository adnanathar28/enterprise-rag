from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal, Protocol, cast
from urllib.parse import urlparse

import httpx
from pydantic import Field, field_validator
from transformers import AutoTokenizer

from brd_knowledge.core.exceptions import (
    GenerationProviderError,
    IncompleteGenerationError,
    MalformedGenerationResponse,
    PromptBudgetExceeded,
)
from brd_knowledge.llm.base import (
    LLMConfiguration,
    LLMGenerationRequest,
    LLMGenerationResponse,
    LLMProvider,
)

TOKENIZER_MODEL = "Qwen/Qwen3-8B"
TOKENIZER_REVISION: Final = "b968826d9c46dd6066d109eabc6255188de91218"


class QwenTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class LocalQwenConfiguration(LLMConfiguration):
    provider: Literal["local_qwen"] = "local_qwen"
    model: Literal["qwen3:8b"] = "qwen3:8b"
    context_window_tokens: int = Field(default=8192, ge=2048, le=40960)
    max_output_tokens: int = Field(default=1024, gt=0, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    seed: int = 42
    timeout_seconds: float = Field(default=180, gt=0)
    base_url: str = "http://127.0.0.1:11434"
    tokenizer_revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"] = TOKENIZER_REVISION
    tokenizer_cache_dir: Path = Path("data/outputs/tokenizer_cache")
    structured_output: Literal[True] = True

    @field_validator("base_url")
    @classmethod
    def require_loopback(cls, value: str) -> str:
        url = urlparse(value)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "::1", "localhost"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("Local Qwen requires a loopback HTTP Ollama URL.")
        return value.rstrip("/")


class LocalQwenProvider(LLMProvider):
    def __init__(
        self,
        configuration: LocalQwenConfiguration | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        tokenizer: QwenTokenizer | None = None,
    ) -> None:
        self._configuration = (configuration or LocalQwenConfiguration()).model_copy(deep=True)
        self._tokenizer = tokenizer
        self._client = httpx.Client(
            base_url=self._configuration.base_url,
            timeout=self._configuration.timeout_seconds,
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        )

    @property
    def configuration(self) -> LocalQwenConfiguration:
        return self._configuration.model_copy(deep=True)

    def close(self) -> None:
        self._client.close()

    def _get_tokenizer(self) -> QwenTokenizer:
        if self._tokenizer is None:
            try:
                self._tokenizer = cast(
                    QwenTokenizer,
                    AutoTokenizer.from_pretrained(
                        TOKENIZER_MODEL,
                        revision=TOKENIZER_REVISION,
                        cache_dir=str(self._configuration.tokenizer_cache_dir),
                        local_files_only=True,
                        trust_remote_code=False,
                    ),
                )
            except Exception as exc:
                raise GenerationProviderError(
                    "Pinned Qwen tokenizer is unavailable; run the README tokenizer setup command."
                ) from exc
        return self._tokenizer

    def _render(self, request: LLMGenerationRequest) -> str:
        # raw=True below ensures Ollama does not add its own chat template.
        # The schema is a decoding grammar, not additional prompt text.
        return self._get_tokenizer().apply_chat_template(
            [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def count_tokens(self, request: LLMGenerationRequest) -> int:
        return len(self._get_tokenizer().encode(self._render(request), add_special_tokens=False))

    def generate_structured(self, request: LLMGenerationRequest) -> LLMGenerationResponse:
        config = self._configuration
        prompt = self._render(request)
        counted = len(self._get_tokenizer().encode(prompt, add_special_tokens=False))
        if (
            request.max_output_tokens > config.max_output_tokens
            or counted + request.max_output_tokens + config.safety_margin_tokens
            > config.context_window_tokens
        ):
            raise PromptBudgetExceeded("Local Qwen request exceeds the configured token budget.")
        try:
            # Resolve the installed digest, without pulling or loading a model.
            tags = self._client.get("/api/tags")
            tags.raise_for_status()
            models = tags.json().get("models", [])
            digest = next((m.get("digest") for m in models if m.get("name") == config.model), None)
            if not isinstance(digest, str) or not digest:
                raise GenerationProviderError("Configured Qwen model is not installed in Ollama.")
            response = self._client.post(
                "/api/generate",
                json={
                    "model": config.model,
                    "prompt": prompt,
                    "raw": True,
                    "think": False,
                    "stream": False,
                    "format": request.response_schema,
                    "keep_alive": "5m",
                    "options": {
                        "num_ctx": config.context_window_tokens,
                        "num_predict": request.max_output_tokens,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "min_p": config.min_p,
                        "seed": config.seed,
                    },
                },
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise GenerationProviderError("Local Ollama request failed.") from exc
        except (ValueError, TypeError, AttributeError) as exc:
            raise MalformedGenerationResponse(
                "Ollama returned a malformed response envelope."
            ) from exc
        if not isinstance(body, dict):
            raise MalformedGenerationResponse("Ollama returned a non-object response envelope.")
        if body.get("error"):
            raise GenerationProviderError("Ollama reported a generation error.")
        if body.get("done") is not True or body.get("done_reason") != "stop":
            raise IncompleteGenerationError("Local Qwen did not finish within the output budget.")
        actual_count = body.get("prompt_eval_count")
        # Never accept an answer after silent truncation or tokenizer/template drift.
        if type(actual_count) is not int or actual_count != counted:
            raise GenerationProviderError("Ollama prompt count differs from local preflight.")
        output_count = body.get("eval_count")
        if type(output_count) is not int or output_count < 0:
            raise MalformedGenerationResponse("Ollama returned invalid output usage.")
        try:
            payload = json.loads(body.get("response", ""))
        except (ValueError, TypeError) as exc:
            raise MalformedGenerationResponse("Local Qwen returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise MalformedGenerationResponse("Local Qwen returned a non-object JSON answer.")
        return LLMGenerationResponse(
            payload=payload,
            prompt_tokens=actual_count,
            output_tokens=output_count,
            model_version=digest,
        )
