"""Unit tests for eovs.api -- FastAPI endpoints."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from eovs.api import app
from eovs.search import SearchResult


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Plain TestClient -- lifespan is NOT triggered."""
    return TestClient(app, raise_server_exceptions=False)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_health_endpoint_not_loaded(client: TestClient) -> None:
    """GET /health returns 200 with model_loaded=False."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_search_endpoint_fails_if_not_loaded(
    client: TestClient,
) -> None:
    """POST /search returns 503 when searcher_instance is None."""
    response = client.post(
        "/search",
        json={"query": "forest", "top_k": 5},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_search_endpoint_success(client: TestClient) -> None:
    """POST /search returns 200 with results when model is loaded."""
    fake_results = [
        SearchResult(
            id=0,
            filename="1.jpg",
            class_name="River",
            score=0.99,
        ),
        SearchResult(
            id=1,
            filename="2.jpg",
            class_name="Forest",
            score=0.85,
        ),
    ]

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = fake_results

    with patch("eovs.api.searcher_instance", mock_searcher):
        response = client.post(
            "/search",
            json={"query": "forest", "top_k": 2},
        )

    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert "latency_ms" in body
    assert len(body["results"]) == 2
    assert body["results"][0]["filename"] == "1.jpg"
    assert body["query"] == "forest"


def test_search_invalid_payload_empty_query(
    client: TestClient,
) -> None:
    """POST /search with a too-short query returns 422."""
    response = client.post(
        "/search",
        json={"query": "", "top_k": 5},
    )

    assert response.status_code == 422


def test_search_invalid_payload_negative_top_k(
    client: TestClient,
) -> None:
    """POST /search with top_k < 1 returns 422."""
    response = client.post(
        "/search",
        json={"query": "forest", "top_k": -1},
    )

    assert response.status_code == 422
