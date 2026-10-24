"""API smoke tests for Phase 0–1."""

from fastapi.testclient import TestClient

from app import __version__
from app.main import app


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "AI PDF Assistant"
    assert payload["version"] == __version__


def test_health_response_schema() -> None:
    response = client.get("/health")
    assert set(response.json().keys()) == {"status", "app", "version"}


def test_docs_available() -> None:
    response = client.get("/docs")
    assert response.status_code == 200
