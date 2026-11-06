"""Regression tests for overview-aware retrieval and threshold defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.config import Settings
from app.services.overview import (
    is_explicit_summary_request,
    is_overview_question,
    should_use_small_doc_overview,
)
from app.services.rag import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    RAGService,
)
from app.services.vector_store import SearchHit


def _hit(
    text: str,
    *,
    score: float = 0.9,
    doc_id: str = "d1",
    page: int = 1,
    chunk_index: int = 0,
    filename: str = "doc.pdf",
    retrieval: str = "similarity",
) -> SearchHit:
    return SearchHit(
        score=score,
        text=text,
        doc_id=doc_id,
        chunk_id=f"{doc_id}:p{page}:c{chunk_index}",
        chunk_index=chunk_index,
        page=page,
        filename=filename,
        retrieval=retrieval,  # type: ignore[arg-type]
    )


@dataclass
class FakeIngest:
    hits: list[SearchHit]
    overview: list[SearchHit] | None = None
    chunk_count: int = 3
    doc_count: int = 1

    def search(self, question: str, *, top_k: int | None = None, doc_id: str | None = None):
        if doc_id is not None and doc_id == "missing":
            raise KeyError(f"Unknown document_id: {doc_id}")
        rows = list(self.hits)
        if doc_id is not None:
            rows = [h for h in rows if h.doc_id == doc_id]
        return rows[: top_k or 5]

    def document_chunk_count(self, doc_id: str | None = None) -> int:
        if doc_id is not None and doc_id == "missing":
            raise KeyError(f"Unknown document_id: {doc_id}")
        return self.chunk_count

    def indexed_document_count(self) -> int:
        return self.doc_count

    def overview_chunks(
        self, *, doc_id: str | None = None, limit: int | None = None
    ) -> list[SearchHit]:
        if doc_id is not None and doc_id == "missing":
            raise KeyError(f"Unknown document_id: {doc_id}")
        source = list(self.overview) if self.overview is not None else list(self.hits)
        if doc_id is not None:
            source = [h for h in source if h.doc_id == doc_id]
        source = sorted(source, key=lambda h: (h.filename, h.chunk_index))
        return source[: limit or 5]


def _settings(**overrides: Any) -> Settings:
    base = {
        "llm_provider": "openai",
        "openai_api_key": "sk-test",
        "similarity_threshold": 0.18,
        "overview_max_chunks": 5,
        "overview_small_doc_max_chunks": 8,
        "overview_weak_score_floor": 0.08,
        "top_k": 5,
        "context_token_budget": 2000,
        "ollama_base_url": "http://127.0.0.1:11434",
    }
    base.update(overrides)
    return Settings(**base)


class _FakeProvider:
    name = "openai"

    def __init__(self, answer: str = "Grounded overview.") -> None:
        self.answer = answer
        self.messages: list | None = None

    def complete(self, messages: list, *, temperature: float = 0.0) -> str:
        self.messages = messages
        return self.answer


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the topic of this document?", True),
        ("What is this PDF about?", True),
        ("Summarize the key findings", True),
        ("What is the aim of the project?", True),
        ("Give me an overview", True),
        ("Where does the indigo phoenix nest?", False),
        ("What year is mentioned?", False),
        ("Who is the CEO salary?", False),
    ],
)
def test_overview_intent_detection(question: str, expected: bool) -> None:
    assert is_overview_question(question) is expected


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Summarize this document", True),
        ("Provide a summary of key findings", True),
        ("What is the topic of this document?", False),
        ("Where does the phoenix nest?", False),
    ],
)
def test_explicit_summary_detection(question: str, expected: bool) -> None:
    assert is_explicit_summary_request(question) is expected


def test_default_similarity_threshold_is_018() -> None:
    settings = Settings(
        llm_provider="ollama",
        openai_api_key="",
        ollama_base_url="http://127.0.0.1:11434",
    )
    assert settings.similarity_threshold == pytest.approx(0.18)


def test_similarity_threshold_rejects_zero() -> None:
    with pytest.raises(Exception):
        Settings(
            llm_provider="ollama",
            openai_api_key="",
            ollama_base_url="http://127.0.0.1:11434",
            similarity_threshold=0.0,
        )


def test_small_doc_overview_gate_conservative() -> None:
    weak = [_hit("abstract text", score=0.12)]
    assert should_use_small_doc_overview(
        is_overview=True,
        chunk_count=5,
        small_doc_max_chunks=8,
        semantic_hits=weak,
        similarity_threshold=0.18,
        weak_score_floor=0.08,
        document_scoped=False,
    )
    # Document-scoped topic questions may use overview even with tiny scores
    assert should_use_small_doc_overview(
        is_overview=True,
        chunk_count=5,
        small_doc_max_chunks=8,
        semantic_hits=[_hit("x", score=-0.05)],
        similarity_threshold=0.18,
        weak_score_floor=0.08,
        document_scoped=True,
    )
    # Not for arbitrary questions
    assert not should_use_small_doc_overview(
        is_overview=False,
        chunk_count=5,
        small_doc_max_chunks=8,
        semantic_hits=weak,
        similarity_threshold=0.18,
        weak_score_floor=0.08,
        document_scoped=True,
    )
    # Not for large docs
    assert not should_use_small_doc_overview(
        is_overview=True,
        chunk_count=20,
        small_doc_max_chunks=8,
        semantic_hits=weak,
        similarity_threshold=0.18,
        weak_score_floor=0.08,
        document_scoped=True,
    )
    # Unscoped: not when scores are too weak
    assert not should_use_small_doc_overview(
        is_overview=True,
        chunk_count=5,
        small_doc_max_chunks=8,
        semantic_hits=[_hit("x", score=0.01)],
        similarity_threshold=0.18,
        weak_score_floor=0.08,
        document_scoped=False,
    )


def test_topic_question_uses_small_doc_overview_when_weak() -> None:
    overview = [
        _hit(
            "Abstract: This paper presents an autonomous drone navigation system.",
            score=0.0,
            chunk_index=0,
            retrieval="overview",
        )
    ]
    weak_semantic = [_hit("unrelated token soup", score=0.12, chunk_index=2)]
    ingest = FakeIngest(hits=weak_semantic, overview=overview, chunk_count=4)
    rag = RAGService(_settings(), ingest)  # type: ignore[arg-type]
    hits = rag.select_hits("What is the topic of this document?", document_id="d1")
    assert hits
    assert "drone" in hits[0].text.lower()
    assert hits[0].retrieval == "overview"
    assert hits[0].score != 1.0


def test_explicit_summary_prefers_overview_over_similarity() -> None:
    overview = [
        _hit(
            "Document lead: drones survey farmland.",
            score=0.0,
            chunk_index=0,
            retrieval="overview",
        )
    ]
    strong_semantic = [
        _hit("Battery voltage peaks at 12.6V", score=0.95, chunk_index=4)
    ]
    ingest = FakeIngest(hits=strong_semantic, overview=overview, chunk_count=6)
    rag = RAGService(_settings(), ingest)  # type: ignore[arg-type]
    hits = rag.select_hits("Summarize this document", document_id="d1")
    assert hits
    assert "drone" in hits[0].text.lower() or "farmland" in hits[0].text.lower()
    assert hits[0].retrieval == "overview"


def test_factual_question_does_not_use_overview_fallback() -> None:
    overview = [
        _hit(
            "Abstract about drones",
            score=0.0,
            chunk_index=0,
            retrieval="overview",
        )
    ]
    weak = [_hit("voltage peaks", score=0.12, chunk_index=2)]
    ingest = FakeIngest(hits=weak, overview=overview, chunk_count=4)
    rag = RAGService(_settings(), ingest)  # type: ignore[arg-type]
    hits = rag.select_hits("What is the battery voltage?", document_id="d1")
    assert hits == []


def test_absent_info_still_insufficient_for_factual() -> None:
    provider = _FakeProvider()
    ingest = FakeIngest(hits=[_hit("drones survey crops", score=0.05)], chunk_count=3)
    rag = RAGService(_settings(), ingest)  # type: ignore[arg-type]
    result = rag.ask(
        "What is the CEO salary mentioned?",
        document_id="d1",
        llm_provider=provider,
    )
    assert result.status == STATUS_INSUFFICIENT
    assert provider.messages is None


def test_unrelated_question_insufficient() -> None:
    ingest = FakeIngest(
        hits=[_hit("autonomous drone abstract", score=0.04)],
        overview=[
            _hit(
                "autonomous drone abstract",
                score=0.0,
                retrieval="overview",
            )
        ],
        chunk_count=2,
    )
    rag = RAGService(_settings(), ingest)  # type: ignore[arg-type]
    result = rag.ask(
        "What is the capital of France?",
        llm_provider=_FakeProvider(),
    )
    assert result.status == STATUS_INSUFFICIENT


def test_single_doc_filter_preserved_on_overview() -> None:
    overview = [
        _hit(
            "alpha abstract",
            score=0.0,
            doc_id="doc-a",
            filename="a.pdf",
            retrieval="overview",
        ),
        _hit(
            "beta abstract",
            score=0.0,
            doc_id="doc-b",
            filename="b.pdf",
            chunk_index=0,
            retrieval="overview",
        ),
    ]
    ingest = FakeIngest(hits=[], overview=overview, chunk_count=2)
    rag = RAGService(_settings(), ingest)  # type: ignore[arg-type]
    hits = rag.select_hits("Summarize the document", document_id="doc-a")
    assert hits
    assert all(h.doc_id == "doc-a" for h in hits)
    assert all(h.retrieval == "overview" for h in hits)


def test_multi_doc_overview_without_filter() -> None:
    overview = [
        _hit(
            "alpha abstract",
            score=0.0,
            doc_id="doc-a",
            filename="a.pdf",
            retrieval="overview",
        ),
        _hit(
            "beta abstract",
            score=0.0,
            doc_id="doc-b",
            filename="b.pdf",
            retrieval="overview",
        ),
    ]
    ingest = FakeIngest(hits=[], overview=overview, chunk_count=4)
    # Empty semantic + weak floor path needs some weak score for small-doc path;
    # explicit summary uses overview regardless.
    rag = RAGService(_settings(), ingest)  # type: ignore[arg-type]
    hits = rag.select_hits("Summarize the key findings")
    assert len(hits) >= 1
    assert all(h.retrieval == "overview" for h in hits)


def test_ask_overview_with_mocked_openai() -> None:
    overview = [
        _hit(
            "This study designs a lightweight drone for abstract mapping.",
            score=0.0,
            retrieval="overview",
        )
    ]
    ingest = FakeIngest(
        hits=[_hit("noise", score=0.11)],
        overview=overview,
        chunk_count=3,
    )
    provider = _FakeProvider("The topic is lightweight drone mapping.")
    rag = RAGService(_settings(llm_provider="openai"), ingest)  # type: ignore[arg-type]
    result = rag.ask(
        "What is the topic of this document?",
        llm_provider=provider,
    )
    assert result.status == STATUS_OK
    assert result.sources
    assert result.sources[0].retrieval == "overview"
    assert "drone" in provider.messages[1]["content"].lower()  # type: ignore[index]
    assert "retrieval=overview" in provider.messages[1]["content"]  # type: ignore[index]


def test_ask_overview_with_mocked_ollama() -> None:
    overview = [
        _hit(
            "Abstract Drone navigation research overview.",
            score=0.0,
            retrieval="overview",
        )
    ]
    ingest = FakeIngest(
        hits=[_hit("noise", score=0.10)],
        overview=overview,
        chunk_count=2,
    )
    provider = _FakeProvider("Topic: drone navigation.")
    provider.name = "ollama"
    rag = RAGService(
        _settings(llm_provider="ollama", openai_api_key=""),
        ingest,  # type: ignore[arg-type]
    )
    result = rag.ask(
        "What is this document about?",
        llm_provider=provider,
    )
    assert result.status == STATUS_OK
    assert result.sources[0].retrieval == "overview"
    assert "drone" in result.answer.lower()
