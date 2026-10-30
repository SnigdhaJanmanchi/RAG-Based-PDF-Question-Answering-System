"""Tests for the Streamlit FastAPI client (mocked HTTP)."""

from __future__ import annotations

import httpx
import pytest

from frontend.api_client import (
    APIClient,
    APIClientError,
    resolve_api_base_url,
)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_resolve_api_base_url_strips_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)
    assert resolve_api_base_url("http://localhost:8000/") == "http://localhost:8000"
    monkeypatch.setenv("API_BASE_URL", "http://api:9000/")
    assert resolve_api_base_url() == "http://api:9000"


def test_health_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "app": "AI PDF Assistant", "version": "0.1.0"})

    with APIClient(base_url="http://test", transport=_mock_transport(handler)) as client:
        assert client.health()["status"] == "ok"


def test_list_documents() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/documents"
        return httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "document_id": "d1",
                        "filename": "a.pdf",
                        "page_count": 2,
                        "chunk_count": 4,
                    }
                ]
            },
        )

    with APIClient(base_url="http://test", transport=_mock_transport(handler)) as client:
        docs = client.list_documents()
        assert len(docs) == 1
        assert docs[0].document_id == "d1"
        assert docs[0].filename == "a.pdf"


def test_upload_pdf_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/documents/upload"
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "document_id": "d1",
                "filename": "sample.pdf",
                "page_count": 2,
                "chunk_count": 3,
                "duplicate": False,
            },
        )

    with APIClient(base_url="http://test", transport=_mock_transport(handler)) as client:
        result = client.upload_pdf(filename="sample.pdf", content=b"%PDF-1.4")
        assert result.document_id == "d1"
        assert result.chunk_count == 3


def test_upload_maps_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "OCR is not yet supported"})

    with APIClient(base_url="http://test", transport=_mock_transport(handler)) as client:
        with pytest.raises(APIClientError) as exc:
            client.upload_pdf(filename="blank.pdf", content=b"%PDF")
        assert "OCR" in exc.value.message
        assert exc.value.status_code == 422


def test_ask_passes_document_id() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = httpx.Request("POST", request.url, content=request.content).content
        import json

        payload = json.loads(request.content.decode())
        seen["payload"] = payload
        return httpx.Response(
            200,
            json={
                "question": payload["question"],
                "answer": "Lake Azure",
                "status": "ok",
                "sources": [
                    {
                        "document_id": "d1",
                        "filename": "sample.pdf",
                        "page": 1,
                        "chunk_id": "d1:p1:c0",
                        "excerpt": "indigo phoenix",
                        "score": 0.9,
                    }
                ],
                "documents": [{"document_id": "d1", "filename": "sample.pdf"}],
            },
        )

    with APIClient(base_url="http://test", transport=_mock_transport(handler)) as client:
        result = client.ask("Where?", document_id="d1", top_k=3)
        assert result.status == "ok"
        assert result.sources[0].page == 1
        assert seen["payload"]["document_id"] == "d1"
        assert seen["payload"]["top_k"] == 3


def test_ask_insufficient_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "question": "???",
                "answer": "I could not find enough relevant information…",
                "status": "insufficient_evidence",
                "sources": [],
                "documents": [],
            },
        )

    with APIClient(base_url="http://test", transport=_mock_transport(handler)) as client:
        result = client.ask("???")
        assert result.status == "insufficient_evidence"
        assert result.sources == []


def test_ask_missing_key_503() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"detail": "OPENAI_API_KEY is not configured. Set it in the environment or .env to use POST /ask."},
        )

    with APIClient(base_url="http://test", transport=_mock_transport(handler)) as client:
        with pytest.raises(APIClientError) as exc:
            client.ask("Hi")
        assert exc.value.status_code == 503
        assert "OPENAI_API_KEY" in exc.value.message


def test_connection_error_is_user_safe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with APIClient(base_url="http://test", transport=_mock_transport(handler)) as client:
        with pytest.raises(APIClientError) as exc:
            client.health()
        assert "Cannot reach the API" in exc.value.message
        assert "Traceback" not in exc.value.message
