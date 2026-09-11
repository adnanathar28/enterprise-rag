from unittest.mock import Mock

from brd_knowledge.api import dependencies
from brd_knowledge.core.config import Settings


def test_question_service_uses_configured_production_reranker(monkeypatch) -> None:
    settings = Settings(_env_file=None, RERANKER_CANDIDATE_K=37)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    session = Mock()
    embedding_provider = Mock()
    scorer = Mock()
    dense_retriever = Mock()
    dense_factory = Mock(return_value=dense_retriever)
    monkeypatch.setattr(dependencies, "PgVectorRetriever", dense_factory)
    reranking_retriever = Mock()
    reranking_factory = Mock(return_value=reranking_retriever)
    monkeypatch.setattr(dependencies, "CrossEncoderRerankingRetriever", reranking_factory)

    service = dependencies.question_answering_service_dependency(
        session,
        embedding_provider,
        scorer,
    )

    dense_factory.assert_called_once_with(session, embedding_provider)
    reranking_factory.assert_called_once_with(
        dense_retriever,
        scorer,
        candidate_k=37,
    )
    assert service._retriever is reranking_retriever
