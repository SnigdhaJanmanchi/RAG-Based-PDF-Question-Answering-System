"""End-to-end API flows mirroring Streamlit: upload → list → select → search/ask."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.config import get_settings
from frontend.api_client import APIClient, DocumentInfo
from frontend.ui_helpers import (
    ALL_DOCUMENTS_LABEL,
    document_choice_labels,
    document_id_for_label,
    file_content_digest,
    should_skip_upload,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_upload_list_select_search_flow(client) -> None:
    """Streamlit-equivalent flow via FastAPI TestClient + UI selection helpers."""
    sample = (FIXTURES / "sample.pdf").read_bytes()
    multi = (FIXTURES / "multipage_facts.pdf").read_bytes()

    r1 = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample, "application/pdf")},
    )
    r2 = client.post(
        "/documents/upload",
        files={"file": ("multipage_facts.pdf", multi, "application/pdf")},
    )
    assert r1.status_code == 200 and r2.status_code == 200
    body1, body2 = r1.json(), r2.json()
    assert body1["page_count"] == 2
    assert body2["page_count"] == 3

    listed = client.get("/documents").json()["documents"]
    docs = [
        DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            page_count=d["page_count"],
            chunk_count=d["chunk_count"],
        )
        for d in listed
    ]
    assert len(docs) >= 2
    labels = document_choice_labels(docs)
    assert labels[0] == ALL_DOCUMENTS_LABEL
    sample_label = next(lbl for lbl in labels if "sample.pdf" in lbl)
    doc_id = document_id_for_label(sample_label, docs)
    assert doc_id == body1["document_id"]

    dup = client.post(
        "/documents/upload",
        files={"file": ("sample-again.pdf", sample, "application/pdf")},
    ).json()
    assert dup["duplicate"] is True
    assert dup["document_id"] == body1["document_id"]

    digest = file_content_digest(sample)
    assert should_skip_upload(digest, {digest}) is True

    search = client.post(
        "/search",
        json={
            "query": "Where does the indigo phoenix nest?",
            "document_id": doc_id,
            "top_k": 3,
        },
    )
    assert search.status_code == 200
    hits = search.json()["results"]
    assert hits
    assert hits[0]["filename"] == "sample.pdf"
    assert hits[0]["page"] == 1


def test_frontend_api_client_against_mock_transport_matches_contracts() -> None:
    """Sanity: Streamlit APIClient still maps real response shapes (mocked HTTP)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/documents":
            return httpx.Response(
                200,
                json={
                    "documents": [
                        {
                            "document_id": "d1",
                            "filename": "a.pdf",
                            "page_count": 1,
                            "chunk_count": 2,
                        }
                    ]
                },
            )
        if request.url.path == "/documents/upload":
            return httpx.Response(
                200,
                json={
                    "document_id": "d1",
                    "filename": "a.pdf",
                    "page_count": 1,
                    "chunk_count": 2,
                    "duplicate": False,
                },
            )
        return httpx.Response(404, json={"detail": "missing"})

    with APIClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as api:
        docs = api.list_documents()
        assert docs[0].filename == "a.pdf"
        up = api.upload_pdf(filename="a.pdf", content=b"%PDF")
        assert up.document_id == "d1"


def test_doc_specific_vs_cross_doc_search(client, sample_pdf_bytes: bytes) -> None:
    sample = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    ).json()
    second = client.post(
        "/documents/upload",
        files={
            "file": (
                "second_doc.pdf",
                (FIXTURES / "second_doc.pdf").read_bytes(),
                "application/pdf",
            )
        },
    ).json()

    scoped = client.post(
        "/search",
        json={
            "query": "Where does the cerulean otter hibernate?",
            "document_id": second["document_id"],
            "top_k": 3,
        },
    )
    assert scoped.status_code == 200
    results = scoped.json()["results"]
    assert results
    assert all(r["document_id"] == second["document_id"] for r in results)
    assert any("Kelp Bay" in r["text"] for r in results)

    wrong = client.post(
        "/search",
        json={
            "query": "Where does the cerulean otter hibernate?",
            "document_id": sample["document_id"],
            "top_k": 3,
        },
    )
    assert wrong.status_code == 200
    wrong_blob = " ".join(r["text"] for r in wrong.json()["results"])
    assert "Kelp Bay" not in wrong_blob


def test_persistence_survives_store_reload(tmp_path, monkeypatch) -> None:
    from app.config import Settings
    from app.services.embeddings import get_embedding_service
    from app.services.ingest import IngestService
    from app.services.vector_store import VectorStore

    data = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data))
    get_settings.cache_clear()
    settings = Settings(data_dir=str(data), chunk_size=300, chunk_overlap=50, top_k=3)
    emb = get_embedding_service(settings.embedding_model)
    store1 = VectorStore(settings.data_dir, dimension=emb.dimension)
    ingest1 = IngestService(settings, emb, store1)
    up = ingest1.ingest_pdf(
        (FIXTURES / "sample.pdf").read_bytes(),
        filename="sample.pdf",
        content_type="application/pdf",
    )
    hits1 = ingest1.search("indigo phoenix Lake Azure", top_k=2, doc_id=up.document_id)
    assert hits1
    assert hits1[0].page == 1

    store2 = VectorStore(settings.data_dir, dimension=emb.dimension)
    ingest2 = IngestService(settings, emb, store2)
    docs = ingest2.store.list_documents()
    assert any(d.doc_id == up.document_id for d in docs)
    hits2 = ingest2.search("indigo phoenix Lake Azure", top_k=2, doc_id=up.document_id)
    assert hits2
    assert hits2[0].filename == "sample.pdf"
    assert "Lake Azure" in hits2[0].text or "phoenix" in hits2[0].text.lower()
    get_settings.cache_clear()


def test_ask_citations_match_retrieved_sources(
    client, sample_pdf_bytes: bytes, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.1")
    get_settings.cache_clear()

    def fake_chat(client_obj, *, model, messages, temperature=0.0):
        return "The indigo phoenix nests at Lake Azure (sample.pdf, page 1)."

    monkeypatch.setattr("app.services.rag.chat_completion", fake_chat)
    monkeypatch.setattr("app.services.rag.build_openai_client", lambda **kw: object())

    doc_id = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    ).json()["document_id"]

    search = client.post(
        "/search",
        json={"query": "Where does the indigo phoenix nest?", "document_id": doc_id},
    ).json()["results"]
    ask = client.post(
        "/ask",
        json={"question": "Where does the indigo phoenix nest?", "document_id": doc_id},
    ).json()

    assert ask["status"] == "ok"
    assert ask["sources"]
    search_chunk_ids = {h["chunk_id"] for h in search}
    for src in ask["sources"]:
        assert src["document_id"] == doc_id
        assert src["filename"] == "sample.pdf"
        assert src["page"] >= 1
        assert src["chunk_id"] in search_chunk_ids or any(
            src["excerpt"][:40] in h["text"] for h in search
        )
    get_settings.cache_clear()


def test_gitignore_excludes_secrets_and_runtime_data() -> None:
    # Resolve relative to repo root so CI runners (not only /workspace) work.
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "data/" in gitignore or "data/*" in gitignore
