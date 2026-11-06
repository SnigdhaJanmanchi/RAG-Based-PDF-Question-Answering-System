"""Unit tests for Streamlit UI helpers (no Streamlit runtime required)."""

from __future__ import annotations

from frontend.api_client import AskResult, AskSource, DocumentInfo
from frontend.ui_helpers import (
    ALL_DOCUMENTS_LABEL,
    assistant_message_from_ask,
    document_choice_labels,
    document_id_for_label,
    file_content_digest,
    format_source_markdown,
    is_insufficient,
    selection_caption,
    should_skip_upload,
)


def test_upload_dedupe_by_digest() -> None:
    digest = file_content_digest(b"%PDF-1.4 sample")
    assert should_skip_upload(digest, set()) is False
    assert should_skip_upload(digest, {digest}) is True


def test_document_selection_all_and_specific() -> None:
    docs = [
        DocumentInfo("id-a", "a.pdf", 2, 5),
        DocumentInfo("id-b", "b.pdf", 1, 2),
    ]
    labels = document_choice_labels(docs)
    assert labels[0] == ALL_DOCUMENTS_LABEL
    assert document_id_for_label(ALL_DOCUMENTS_LABEL, docs) is None
    assert document_id_for_label(labels[1], docs) == "id-a"
    assert document_id_for_label(labels[2], docs) == "id-b"


def test_selection_caption_empty_and_scoped() -> None:
    assert "empty" in selection_caption(None, []).lower() or "upload" in selection_caption(None, []).lower()
    docs = [DocumentInfo("id-a", "report.pdf", 3, 9)]
    assert "all indexed" in selection_caption(None, docs).lower()
    assert "report.pdf" in selection_caption("id-a", docs)


def test_assistant_message_and_citations() -> None:
    result = AskResult(
        question="Where?",
        answer="Lake Azure",
        status="ok",
        sources=[
            AskSource("d1", "sample.pdf", 1, "d1:p1:c0", "indigo phoenix nests", 0.88)
        ],
        documents=[{"document_id": "d1", "filename": "sample.pdf"}],
    )
    msg = assistant_message_from_ask(result)
    assert msg["role"] == "assistant"
    assert msg["sources"][0]["page"] == 1
    assert msg["sources"][0]["filename"] == "sample.pdf"
    assert msg["sources"][0]["retrieval"] == "similarity"
    md = format_source_markdown(result.sources[0])
    assert "sample.pdf" in md
    assert "page 1" in md
    assert "similarity 0.880" in md
    assert "Document overview" not in md


def test_overview_citation_labeled_not_as_similarity_score() -> None:
    source = AskSource(
        "d1",
        "Abstract Drone.pdf",
        1,
        "d1:p1:c0",
        "autonomous drone navigation abstract",
        0.0,
        retrieval="overview",
    )
    md = format_source_markdown(source)
    assert "Document overview" in md
    assert "similarity" not in md
    assert "score" not in md.lower()
    # Must not present overview as a cosine of 1.0
    assert "1.000" not in md
    assert "0.000" not in md


def test_insufficient_flag() -> None:
    assert is_insufficient("insufficient_evidence") is True
    assert is_insufficient("ok") is False
