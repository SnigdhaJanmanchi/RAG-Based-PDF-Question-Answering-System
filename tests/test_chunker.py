"""Unit tests for recursive chunking."""

from __future__ import annotations

import pytest

from app.services.chunker import chunk_pages
from app.services.pdf_parser import PageText


def test_chunk_metadata_includes_doc_page_and_id() -> None:
    pages = [
        PageText(page_number=1, text="Alpha " * 50),
        PageText(page_number=2, text="Beta " * 50),
    ]
    chunks = chunk_pages(
        pages,
        doc_id="doc-1",
        filename="sample.pdf",
        chunk_size=80,
        chunk_overlap=20,
    )
    assert chunks
    assert all(c.doc_id == "doc-1" for c in chunks)
    assert all(c.filename == "sample.pdf" for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(f"doc-1:p{c.page}:c{c.chunk_index}" == c.chunk_id for c in chunks)


def test_chunk_skips_empty_pages_but_keeps_traceability() -> None:
    pages = [
        PageText(page_number=1, text=""),
        PageText(page_number=2, text="Only page two has content about lakes."),
    ]
    chunks = chunk_pages(
        pages,
        doc_id="d",
        filename="f.pdf",
        chunk_size=100,
        chunk_overlap=10,
    )
    assert len(chunks) >= 1
    assert all(c.page == 2 for c in chunks)


def test_invalid_chunk_params() -> None:
    pages = [PageText(page_number=1, text="hello world")]
    with pytest.raises(ValueError):
        chunk_pages(
            pages, doc_id="d", filename="f.pdf", chunk_size=10, chunk_overlap=10
        )
