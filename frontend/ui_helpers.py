"""Pure helpers for Streamlit UI state (easy to unit-test)."""

from __future__ import annotations

import hashlib
from typing import Any

from frontend.api_client import AskResult, AskSource, DocumentInfo

ALL_DOCUMENTS_LABEL = "All indexed documents"
EXAMPLE_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key points on the first page.",
    "Where does the indigo phoenix nest?",
]


def file_content_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def should_skip_upload(digest: str, already_processed: set[str]) -> bool:
    """Avoid re-uploading the same bytes on Streamlit reruns."""
    return digest in already_processed


def document_choice_labels(documents: list[DocumentInfo]) -> list[str]:
    labels = [ALL_DOCUMENTS_LABEL]
    for doc in documents:
        labels.append(
            f"{doc.filename}  ({doc.page_count} pages · {doc.chunk_count} chunks)"
        )
    return labels


def document_id_for_label(
    label: str, documents: list[DocumentInfo]
) -> str | None:
    if label == ALL_DOCUMENTS_LABEL or not label:
        return None
    for doc in documents:
        prefix = f"{doc.filename}  ("
        if label.startswith(prefix) or label.startswith(doc.filename):
            # Prefer exact filename match when labels are unique enough.
            if doc.filename in label and doc.document_id:
                # Disambiguate by matching full constructed label when possible.
                expected = (
                    f"{doc.filename}  ({doc.page_count} pages · {doc.chunk_count} chunks)"
                )
                if label == expected:
                    return doc.document_id
    # Fallback: first filename prefix match
    for doc in documents:
        if label.startswith(doc.filename):
            return doc.document_id
    return None


def selection_caption(document_id: str | None, documents: list[DocumentInfo]) -> str:
    if not documents:
        return "No documents indexed yet — upload a PDF to get started."
    if document_id is None:
        return "Asking across all indexed documents."
    for doc in documents:
        if doc.document_id == document_id:
            return f"Asking about: {doc.filename}"
    return "Selected document is no longer in the library — choose again."


def format_source_markdown(source: AskSource) -> str:
    excerpt = source.excerpt.strip().replace("\n", " ")
    if len(excerpt) > 400:
        excerpt = excerpt[:397] + "..."
    return (
        f"**{source.filename}** · page {source.page} · score {source.score:.3f}\n\n"
        f"> {excerpt}"
    )


def assistant_message_from_ask(result: AskResult) -> dict[str, Any]:
    """Build a chat message dict stored in session_state."""
    return {
        "role": "assistant",
        "content": result.answer,
        "status": result.status,
        "sources": [
            {
                "document_id": s.document_id,
                "filename": s.filename,
                "page": s.page,
                "chunk_id": s.chunk_id,
                "excerpt": s.excerpt,
                "score": s.score,
            }
            for s in result.sources
        ],
        "documents": result.documents,
    }


def is_insufficient(status: str) -> bool:
    return status == "insufficient_evidence"
