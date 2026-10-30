"""Live E2E against local Ollama (skipped when daemon/model unavailable)."""

from __future__ import annotations

import os
import resource
import time

import httpx
import pytest

from app.config import get_settings

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")


def _ollama_ready() -> bool:
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        response.raise_for_status()
        models = {
            str(m.get("name") or "").strip()
            for m in (response.json().get("models") or [])
        }
        return OLLAMA_MODEL in models
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.live_ollama
def test_ask_live_ollama_pdf_e2e(
    client, sample_pdf_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real upload → retrieve → Ollama generate on sample.pdf."""
    if not _ollama_ready():
        pytest.skip(
            f"Local Ollama at {OLLAMA_URL} with model {OLLAMA_MODEL} not available"
        )

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENAI_API_KEY", "")  # must work without OpenAI
    monkeypatch.setenv("OLLAMA_BASE_URL", OLLAMA_URL)
    monkeypatch.setenv("OLLAMA_MODEL", OLLAMA_MODEL)
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.15")
    get_settings.cache_clear()

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()

    upload = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()["document_id"]

    ask_started = time.perf_counter()
    response = client.post(
        "/ask",
        json={
            "question": "Where does the indigo phoenix nest, and in what year?",
            "document_id": doc_id,
            "top_k": 3,
        },
    )
    ask_ms = (time.perf_counter() - ask_started) * 1000
    total_ms = (time.perf_counter() - t0) * 1000
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["sources"], "expected citations from retrieval"
    assert all(s["filename"] == "sample.pdf" for s in body["sources"])
    lower = body["answer"].lower()
    # Small local model — accept either landmark or bird name as grounded signal.
    assert "azure" in lower or "phoenix" in lower or "1842" in lower
    # Thinking must never appear in the API answer.
    assert "<think>" not in lower
    assert "thinking" not in body

    # Soft resource signals (Linux ru_maxrss is KiB)
    rss_mb = rss_after / 1024.0
    print(
        f"[live_ollama] ask_ms={ask_ms:.0f} total_ms={total_ms:.0f} "
        f"rss_max_mb={rss_mb:.1f} rss_before_kib={rss_before} rss_after_kib={rss_after}"
    )
    # Guardrail: process should not balloon past ~3.5 GiB max RSS in this E2E.
    assert rss_mb < 3500, f"possible OOM risk: max RSS {rss_mb:.0f} MiB"

    get_settings.cache_clear()
