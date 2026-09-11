from __future__ import annotations

import argparse
import json
import sys

from pydantic import ValidationError

from brd_knowledge.core.config import get_settings
from brd_knowledge.core.exceptions import GenerationError, RetrievalRerankingError
from brd_knowledge.database.session import SessionLocal
from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.llm.factory import create_llm_provider
from brd_knowledge.retrieval import (
    CrossEncoderRerankingRetriever,
    PgVectorRetriever,
    TransformersCrossEncoderScorer,
)
from brd_knowledge.schemas.query import DocumentQuestionRequest
from brd_knowledge.services.question_answering_service import QuestionAnsweringService


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer one question using dense retrieval, reranking, and a configured LLM."
    )
    parser.add_argument("question")
    parser.add_argument("--provider", choices=["gemini", "local_qwen"])
    parser.add_argument("--model", help="Override the selected provider model.")
    parser.add_argument(
        "--document-id", required=True, help="Restrict evidence to an approved document."
    )
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument("--max-characters", type=positive_int, default=20_000)
    parser.add_argument("--max-output-tokens", type=positive_int, default=1024)
    args = parser.parse_args()
    if not args.question.strip() or not args.document_id.strip():
        parser.error("Question and document ID must not be blank.")
    return args


def main() -> None:
    args = parse_args()
    settings = get_settings()
    provider = create_llm_provider(settings, provider=args.provider, model=args.model)
    provider_handed_off = False
    try:
        configuration = provider.configuration
        if (
            configuration.max_output_tokens
            and args.max_output_tokens > configuration.max_output_tokens
        ):
            raise ValueError("max-output-tokens exceeds the configured model output limit.")
        embedding_provider = GteModernBertEmbeddingProvider(
            model_name=settings.embedding_model_name,
            model_revision=settings.embedding_model_revision,
            dimension=settings.embedding_dimension,
            max_sequence_length=settings.embedding_max_sequence_length,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            preprocessing_version=settings.embedding_preprocessing_version,
        )
        reranker_scorer = TransformersCrossEncoderScorer(
            model_name=settings.reranker_model_name,
            model_revision=settings.reranker_model_revision,
            max_sequence_length=settings.reranker_max_sequence_length,
            batch_size=settings.reranker_batch_size,
            device=settings.reranker_device,
        )
        with SessionLocal() as session:
            question_service = QuestionAnsweringService(
                CrossEncoderRerankingRetriever(
                    PgVectorRetriever(session, embedding_provider),
                    reranker_scorer,
                    candidate_k=settings.reranker_candidate_k,
                ),
                settings,
                provider_factory=lambda *_args, **_kwargs: provider,
            )
            provider_handed_off = True
            result = question_service.answer(
                args.document_id,
                DocumentQuestionRequest(
                    question=args.question,
                    provider=args.provider,
                    model=args.model,
                ),
                top_k=args.top_k,
                max_context_characters=args.max_characters,
                max_output_tokens=args.max_output_tokens,
            )
    except Exception:
        # The shared service owns provider closure after it starts. Close providers
        # that fail validation before orchestration begins.
        if not provider_handed_off:
            provider.close()
        raise
    print(
        json.dumps(
            {
                "answer": result.answer.model_dump(mode="json"),
                "configuration": configuration.model_dump(mode="json"),
                "context": {
                    "retrieved_chunks": len(result.retrieved_chunks),
                    "included_chunks": len(result.context.evidence),
                    "used_characters": result.context.used_characters,
                    "exclusions": [
                        item.model_dump(mode="json") for item in result.context.exclusions
                    ],
                },
                "elapsed_seconds": round(result.elapsed_seconds, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (GenerationError, RetrievalRerankingError, ValidationError, ValueError) as exc:
        # Do not print SDK exceptions, chained tracebacks, or settings values.
        print(
            f"{type(exc).__name__}: query failed; check configuration and budgets.", file=sys.stderr
        )
        sys.exit(1)
