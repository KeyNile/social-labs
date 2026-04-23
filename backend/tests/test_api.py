import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["API_KEYS"] = "test-key"
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

NARRATIVE_DIR = str(Path(__file__).parent.parent.parent / "output" / "narratives")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    mock_engine = MagicMock()
    mock_engine.search.return_value = [("ID_001", 0.9), ("ID_002", 0.8)]
    mock_anthropic = MagicMock()

    # Remove cached module so patches apply cleanly
    sys.modules.pop("api_server", None)

    with patch("embeddings.EmbeddingEngine", return_value=mock_engine):
        import api_server as server_module
        # Set state directly (bypass lifespan for unit tests)
        server_module.app.state.engine = mock_engine
        server_module.app.state.narrative_dir = NARRATIVE_DIR
        server_module.app.state.anthropic_client = mock_anthropic
        yield TestClient(server_module.app)


def test_query_requires_api_key(client):
    resp = client.post("/v1/query", json={"context": "career change", "n": 3})
    assert resp.status_code in (401, 403)


def test_query_returns_personas(client):
    resp = client.post(
        "/v1/query",
        json={"context": "career change", "n": 2},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "personas" in data
    assert len(data["personas"]) <= 2


def test_simulate_invalid_mode_rejected(client):
    resp = client.post(
        "/v1/simulate",
        json={"persona_ids": ["ID_001"], "question": "test", "mode": "invalid"},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 422


def test_trajectory_404_for_unknown_persona(client):
    resp = client.get(
        "/v1/personas/ID_999/trajectory",
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 404


def test_trajectory_returns_timeline_for_valid_persona(client):
    resp = client.get(
        "/v1/personas/ID_001/trajectory",
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "timeline" in data
    assert "background" in data
    assert len(data["timeline"]) > 0
