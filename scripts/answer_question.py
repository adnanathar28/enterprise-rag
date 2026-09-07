from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter

from pydantic import ValidationError

from brd_knowledge.context import ContextBuilder
from brd_knowledge.core.config import get_settings
from brd_knowledge.core.exceptions import GenerationError
from brd_knowledge.database.session import SessionLocal
from brd_knowledge.embeddings.gte_modernbert import GteModernBertEmbeddingProvider
from brd_knowledge.generation import GroundedAnswerService
from brd_knowledge.llm.factory import create_llm_provider
from brd_knowledge.retrieval import PgVectorRetriever
from brd_knowledge.schemas.context import ContextBuildRequest
from brd_knowledge.schemas.generation import GroundedAnswerRequest


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer one question using dense retrieval and a configured LLM."
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
    try:
        configuration = provider.configuration
        if (
            configuration.max_output_tokens
            and args.max_output_tokens > configuration.max_output_tokens
        ):
            raise ValueError("max-output-tokens exceeds the configured model output limit.")
        started = perf_counter()
        embedding_provider = GteModernBertEmbeddingProvider(
            model_name=settings.embedding_model_name,
            model_revision=settings.embedding_model_revision,
            dimension=settings.embedding_dimension,
            max_sequence_length=settings.embedding_max_sequence_length,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            preprocessing_version=settings.embedding_preprocessing_version,
        )
        with SessionLocal() as session:
            chunks = PgVectorRetriever(session, embedding_provider).search(
                args.question, top_k=args.top_k, document_id=args.document_id
            )
        context = ContextBuilder().build(
            ContextBuildRequest(retrieved_chunks=chunks, max_characters=args.max_characters)
        )
        answer = GroundedAnswerService(provider).generate(
            GroundedAnswerRequest(
                question=args.question, context=context, max_output_tokens=args.max_output_tokens
            )
        )
    finally:
        provider.close()
    print(
        json.dumps(
            {
                "answer": answer.model_dump(mode="json"),
                "configuration": configuration.model_dump(mode="json"),
                "context": {
                    "retrieved_chunks": len(chunks),
                    "included_chunks": len(context.evidence),
                    "used_characters": context.used_characters,
                    "exclusions": [item.model_dump(mode="json") for item in context.exclusions],
                },
                "elapsed_seconds": round(perf_counter() - started, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (GenerationError, ValidationError, ValueError) as exc:
        # Do not print SDK exceptions, chained tracebacks, or settings values.
        print(
            f"{type(exc).__name__}: query failed; check configuration and budgets.", file=sys.stderr
        )
        sys.exit(1)
