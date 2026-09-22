"""Request-scoped, content-safe diagnostics for grounded generation failures."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class GenerationDiagnosticState:
    request_id: str
    document_id: str
    include_payload: bool
    credential: str | None = None
    model: str | None = None
    model_version: str | None = None
    finish_reason: str | None = None
    provider_request_id: str | None = None
    raw_generated_text: str | None = None


_state: ContextVar[GenerationDiagnosticState | None] = ContextVar(
    "generation_diagnostic_state", default=None
)


@contextmanager
def generation_diagnostic_scope(
    document_id: str, *, include_payload: bool, credential: str | None = None
) -> Iterator[None]:
    token = _state.set(
        GenerationDiagnosticState(
            request_id=uuid4().hex,
            document_id=document_id,
            include_payload=include_payload,
            credential=credential,
        )
    )
    try:
        yield
    finally:
        _state.reset(token)


def record_generation_model(model: str) -> None:
    state = _state.get()
    if state is not None:
        state.model = model


def record_gemini_response(
    *,
    model: str,
    model_version: str | None,
    finish_reason: str | None,
    provider_request_id: str | None,
    raw_generated_text: str | None = None,
) -> None:
    state = _state.get()
    if state is None:
        return
    state.model = model
    state.model_version = model_version
    state.finish_reason = finish_reason
    state.provider_request_id = provider_request_id
    if state.include_payload:
        state.raw_generated_text = raw_generated_text


def log_generation_failure(
    stage: str,
    exc: Exception,
    *,
    model: str | None = None,
    payload: Any = None,
) -> None:
    state = _state.get()
    details: dict[str, Any] = {
        "stage": stage,
        "exception_class": type(exc).__name__,
        "exception_message": _safe_message(
            exc, include_payload=bool(state and state.include_payload)
        ),
    }
    if state is not None:
        details.update(
            request_id=state.request_id,
            document_id=state.document_id,
            model=state.model or model,
            model_version=state.model_version,
            finish_reason=state.finish_reason,
            provider_request_id=state.provider_request_id,
        )
        if state.include_payload:
            details["raw_generated_text"] = state.raw_generated_text
            if payload is not None:
                details["parsed_payload"] = payload
    elif model is not None:
        details["model"] = model
    if state is not None and state.credential:
        details = _redact_credential(details, state.credential)
    logger.warning(
        "grounded_generation_failure %s",
        json.dumps(details, ensure_ascii=False, default=str),
    )


def _safe_message(exc: Exception, *, include_payload: bool) -> str:
    message = str(exc)
    if include_payload:
        return message
    if message.startswith("Malformed inline citation group:"):
        return "Malformed inline citation group: [redacted]"
    if message.startswith("Generated answer cites unknown evidence IDs:"):
        return "Generated answer cites unknown evidence IDs: [redacted]"
    return message


def _redact_credential(value: Any, credential: str) -> Any:
    if isinstance(value, str):
        return value.replace(credential, "[redacted credential]")
    if isinstance(value, list):
        return [_redact_credential(item, credential) for item in value]
    if isinstance(value, dict):
        return {
            _redact_credential(key, credential): _redact_credential(item, credential)
            for key, item in value.items()
        }
    return value
