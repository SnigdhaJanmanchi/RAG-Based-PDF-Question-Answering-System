"""Integration tests: real MiniLM+FAISS overview retrieval (mocked LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings

FIXTURES = Path(__file__).parent / "fixtures"
ABSTRACT_DRONE = FIXTURES / "abstract_drone.pdf"
SAMPLE = FIXTURES / "sample.pdf"
CORRUPT = FIXTURES / "corrupt.pdf"
BLANK = FIXTURES / "blank.pdf"


class _CapturingProvider:
    name = "openai"

    def __init__(self) -> None:
        self.last_messages: list | None = None

    def complete(self, messages: list, *, temperature: float = 0.0) -> str:
        self.last_messages = messages
        # Echo a grounded stub from context markers.
        user = messages[1]["content"]
        if "drone" in user.lower() or "Abstract Drone" in user:
            return "The topic is autonomous drone navigation and mapping."
        if "indigo" in user.lower() or "phoenix" in user.lower():
            return "The indigo phoenix nests at Lake Azure in 1842."
        return "Grounded answer from retrieved excerpts."


@pytest.fixture()
def overview_env(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.18")
    monkeypatch.setenv("OVERVIEW_SMALL_DOC_MAX_CHUNKS", "8")
    monkeypatch.setenv("OVERVIEW_WEAK_SCORE_FLOOR", "0.08")
    monkeypatch.setenv("OVERVIEW_MAX_CHUNKS", "5")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()


def _upload(client, path: Path, name: str | None = None) -> dict:
    data = path.read_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": (name or path.name, data, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_topic_question_on_abstract_drone(
    client, overview_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _upload(client, ABSTRACT_DRONE, "Abstract Drone.pdf")
    doc_id = body["document_id"]
    assert body["chunk_count"] <= 8

    provider = _CapturingProvider()

    def _fake_build(settings, **kw):
        return provider

    monkeypatch.setattr("app.services.rag.build_llm_provider", _fake_build)

    response = client.post(
        "/ask",
        json={
            "question": "What is the topic of this document?",
            "document_id": doc_id,
            "top_k": 5,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["sources"], "expected overview/semantic sources"
    assert all(s["document_id"] == doc_id for s in payload["sources"])
    lower = payload["answer"].lower()
    assert "drone" in lower or "navigation" in lower or "mapping" in lower
    assert provider.last_messages is not None
    assert "UNTRUSTED_PDF_EXCERPTS_BEGIN" in provider.last_messages[1]["content"]


def test_factual_still_works_on_sample(
    client, overview_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _upload(client, SAMPLE)
    monkeypatch.setattr(
        "app.services.rag.build_llm_provider",
        lambda settings, **kw: _CapturingProvider(),
    )
    response = client.post(
        "/ask",
        json={
            "question": "Where does the indigo phoenix nest, and in what year?",
            "document_id": body["document_id"],
            "top_k": 3,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["sources"]
    assert "azure" in payload["answer"].lower() or "phoenix" in payload["answer"].lower()


def test_absent_info_insufficient(client, overview_env, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _upload(client, ABSTRACT_DRONE, "Abstract Drone.pdf")
    called = {"n": 0}

    class Boom:
        name = "openai"

        def complete(self, messages, *, temperature=0.0):
            called["n"] += 1
            return "should not run"

    monkeypatch.setattr(
        "app.services.rag.build_llm_provider", lambda settings, **kw: Boom()
    )
    response = client.post(
        "/ask",
        json={
            "question": "What is the CEO annual salary?",
            "document_id": body["document_id"],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert called["n"] == 0


def test_unrelated_question_insufficient(
    client, overview_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _upload(client, ABSTRACT_DRONE, "Abstract Drone.pdf")
    monkeypatch.setattr(
        "app.services.rag.build_llm_provider",
        lambda settings, **kw: _CapturingProvider(),
    )
    response = client.post(
        "/ask",
        json={
            "question": "What is the capital of France?",
            "document_id": body["document_id"],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"


def test_multi_doc_summary_and_single_filter(
    client, overview_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _upload(client, ABSTRACT_DRONE, "Abstract Drone.pdf")
    b = _upload(client, SAMPLE)
    provider = _CapturingProvider()
    monkeypatch.setattr(
        "app.services.rag.build_llm_provider", lambda settings, **kw: provider
    )

    scoped = client.post(
        "/ask",
        json={
            "question": "Summarize the key findings",
            "document_id": a["document_id"],
        },
    )
    assert scoped.status_code == 200
    scoped_body = scoped.json()
    assert scoped_body["status"] == "ok"
    assert all(s["document_id"] == a["document_id"] for s in scoped_body["sources"])

    all_docs = client.post(
        "/ask",
        json={"question": "Summarize the documents"},
    )
    assert all_docs.status_code == 200
    assert all_docs.json()["status"] == "ok"


def test_empty_index_overview_insufficient(
    client, overview_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.services.rag.build_llm_provider",
        lambda settings, **kw: _CapturingProvider(),
    )
    response = client.post(
        "/ask",
        json={"question": "What is the topic of this document?"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"


def test_corrupt_pdf_rejected(client, overview_env) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("corrupt.pdf", CORRUPT.read_bytes(), "application/pdf")},
    )
    assert response.status_code >= 400


def test_blank_pdf_edge(client, overview_env) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("blank.pdf", BLANK.read_bytes(), "application/pdf")},
    )
    # Blank may be rejected or indexed with zero/low text — either is acceptable
    # as long as we do not crash.
    assert response.status_code in {200, 400, 422}


def test_both_providers_mocked_for_overview(
    client, overview_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _upload(client, ABSTRACT_DRONE, "Abstract Drone.pdf")

    for provider_name in ("openai", "ollama"):
        monkeypatch.setenv("LLM_PROVIDER", provider_name)
        if provider_name == "ollama":
            monkeypatch.setenv("OPENAI_API_KEY", "")
        else:
            monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        get_settings.cache_clear()

        captured: dict[str, str] = {}

        class P:
            name = provider_name

            def complete(self, messages, *, temperature=0.0):
                captured["provider"] = provider_name
                return f"Answer via {provider_name}: autonomous drone topic."

        monkeypatch.setattr(
            "app.services.rag.build_llm_provider", lambda settings, **kw: P()
        )
        response = client.post(
            "/ask",
            json={
                "question": "What is this document about?",
                "document_id": body["document_id"],
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["status"] == "ok"
        assert captured["provider"] == provider_name
        get_settings.cache_clear()
