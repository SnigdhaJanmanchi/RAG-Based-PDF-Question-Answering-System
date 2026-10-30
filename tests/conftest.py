"""Shared pytest fixtures for Phase 2 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES / "sample.pdf"
BLANK_PDF = FIXTURES / "blank.pdf"
KNOWN_FACT = "indigo phoenix nests at Lake Azure"
KNOWN_YEAR = "1842"


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate FAISS/uploads under a temp directory and clear settings cache."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("DATA_DIR", str(root))
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "40")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.15")
    monkeypatch.setenv("CONTEXT_TOKEN_BUDGET", "2000")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024))
    # Default: no real key in unit/API suite (tests that need a key set it explicitly).
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # Pin OpenAI for the mocked regression suite. Ollama-specific tests override
    # LLM_PROVIDER=ollama. App default remains ollama via Settings / .env.example.
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    return SAMPLE_PDF.read_bytes()


@pytest.fixture()
def blank_pdf_bytes() -> bytes:
    return BLANK_PDF.read_bytes()


@pytest.fixture()
def client(data_dir: Path):
    """FastAPI TestClient bound to an isolated data dir."""
    from fastapi.testclient import TestClient

    from app.main import app

    # Module-level app retains state across tests — reset per isolated data dir.
    app.state.vector_store = None
    app.state.embedding_service = None
    get_settings.cache_clear()

    with TestClient(app) as test_client:
        yield test_client

    app.state.vector_store = None
    app.state.embedding_service = None
