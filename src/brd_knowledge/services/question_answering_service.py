from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Protocol

from brd_knowledge.context import ContextBuilder
from brd_knowledge.core.config import Settings
from brd_knowledge.generation import GroundedAnswerService
from brd_knowledge.llm.base import LLMProvider
from brd_knowledge.llm.factory import create_llm_provider
from brd_knowledge.schemas.context import ContextBuildRequest
from brd_knowledge.schemas.generation import GroundedAnswerRequest
from brd_knowledge.schemas.query import DocumentQuestionRequest, DocumentQuestionResponse
from brd_knowledge.schemas.retrieval import RetrievedChunk

DEFAULT_TOP_K = 5
DEFAULT_CONTEXT_CHARACTERS = 20_000
DEFAULT_MAX_OUTPUT_TOKENS = 1_024


class DocumentRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]: ...


ProviderFactory = Callable[..., LLMProvider]


class QuestionAnsweringService:
    def __init__(
        self,
        retriever: DocumentRetriever,
        settings: Settings,
        *,
        provider_factory: ProviderFactory = create_llm_provider,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._retriever = retriever
        self._settings = settings
        self._provider_factory = provider_factory
        self._context_builder = context_builder or ContextBuilder()

    def answer(
        self,
        document_id: str,
        request: DocumentQuestionRequest,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_context_characters: int = DEFAULT_CONTEXT_CHARACTERS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> DocumentQuestionResponse:
        started = perf_counter()
        provider = self._provider_factory(
            self._settings,
            provider=request.provider,
            model=request.model,
        )
        try:
            retrieved = self._retriever.search(
                request.question,
                top_k=top_k,
                document_id=document_id,
            )
            context = self._context_builder.build(
                ContextBuildRequest(
                    retrieved_chunks=retrieved,
                    max_characters=max_context_characters,
                )
            )
            answer = GroundedAnswerService(provider).generate(
                GroundedAnswerRequest(
                    question=request.question,
                    context=context,
                    max_output_tokens=max_output_tokens,
                )
            )
        finally:
            provider.close()
        return DocumentQuestionResponse(
            document_id=document_id,
            question=request.question,
            answer=answer,
            context=context,
            retrieved_chunks=retrieved,
            elapsed_seconds=perf_counter() - started,
        )
