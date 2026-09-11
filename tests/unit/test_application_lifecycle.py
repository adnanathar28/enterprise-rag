from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from brd_knowledge.core.exceptions import RetrievalRerankingError
from brd_knowledge.main import app


def test_startup_loads_reranker_once_for_process_lifespan(monkeypatch) -> None:
    scorer = Mock()
    scorer_dependency = Mock(return_value=scorer)
    monkeypatch.setattr("brd_knowledge.main.reranker_scorer_dependency", scorer_dependency)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200

    scorer_dependency.assert_called_once_with()
    scorer.load.assert_called_once_with()


def test_startup_fails_when_reranker_initialization_fails(monkeypatch) -> None:
    scorer = Mock()
    scorer.load.side_effect = OSError("private model cache detail")
    monkeypatch.setattr("brd_knowledge.main.reranker_scorer_dependency", lambda: scorer)

    with pytest.raises(
        RetrievalRerankingError,
        match="failed to initialize during application startup",
    ) as exc_info:
        with TestClient(app):
            pass

    assert isinstance(exc_info.value.__cause__, OSError)
