"""API tests for health, upload, list, search, and invalid PDFs."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import __version__
from tests.conftest import KNOWN_FACT, SAMPLE_PDF


def test_health_returns_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "AI PDF Assistant"
    assert payload["version"] == __version__


def test_health_response_schema(client) -> None:
    response = client.get("/health")
    assert set(response.json().keys()) == {"status", "app", "version"}


def test_docs_available(client) -> None:
    response = client.get("/docs")
    assert response.status_code == 200


def test_upload_rejects_non_pdf(client) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_empty_pdf_named(client) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_rejects_blank_pdf_with_ocr_message(client, blank_pdf_bytes: bytes) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("blank.pdf", blank_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


def test_upload_rejects_oversized(client, sample_pdf_bytes: bytes, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("MAX_UPLOAD_BYTES", "50")
    get_settings.cache_clear()
    response = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 413
    get_settings.cache_clear()


@pytest.mark.slow
def test_upload_list_and_search_e2e(client, sample_pdf_bytes: bytes) -> None:
    """Upload → index → search known fact → verify chunk text + page."""
    upload = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert upload.status_code == 200, upload.text
    body = upload.json()
    assert body["filename"] == "sample.pdf"
    assert body["page_count"] == 2
    assert body["chunk_count"] >= 1
    assert body["duplicate"] is False
    doc_id = body["document_id"]

    listed = client.get("/documents")
    assert listed.status_code == 200
    docs = listed.json()["documents"]
    assert any(d["document_id"] == doc_id for d in docs)

    # Alias path
    dup = client.post(
        "/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert dup.status_code == 200
    assert dup.json()["duplicate"] is True
    assert dup.json()["document_id"] == doc_id

    search = client.post(
        "/search",
        json={"query": "Where does the indigo phoenix nest?", "top_k": 3},
    )
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert results
    top = results[0]
    assert KNOWN_FACT.split()[0] in top["text"].lower() or "phoenix" in top["text"].lower()
    assert "Lake Azure" in top["text"] or "lake azure" in top["text"].lower()
    assert top["page"] == 1
    assert top["document_id"] == doc_id
    assert top["filename"] == "sample.pdf"


@pytest.mark.slow
def test_search_scoped_to_document(client, sample_pdf_bytes: bytes) -> None:
    upload = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    doc_id = upload.json()["document_id"]
    missing = client.post(
        "/search",
        json={"query": "phoenix", "document_id": "does-not-exist"},
    )
    assert missing.status_code == 404

    ok = client.post(
        "/search",
        json={"query": "phoenix", "document_id": doc_id, "top_k": 2},
    )
    assert ok.status_code == 200
    assert all(r["document_id"] == doc_id for r in ok.json()["results"])


def test_fixture_pdf_exists() -> None:
    assert SAMPLE_PDF.is_file()
    assert SAMPLE_PDF.stat().st_size > 0
