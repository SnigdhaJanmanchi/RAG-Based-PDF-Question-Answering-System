"""Unit tests for PDF validation and extraction."""

from __future__ import annotations

import pytest

from app.exceptions import (
    InvalidPDFError,
    NoExtractableTextError,
    OversizedPDFError,
)
from app.services.pdf_parser import extract_pages, validate_pdf_upload


def test_validate_rejects_empty() -> None:
    with pytest.raises(InvalidPDFError, match="empty"):
        validate_pdf_upload(b"", filename="x.pdf", content_type="application/pdf", max_bytes=1000)


def test_validate_rejects_oversized(sample_pdf_bytes: bytes) -> None:
    with pytest.raises(OversizedPDFError):
        validate_pdf_upload(
            sample_pdf_bytes,
            filename="sample.pdf",
            content_type="application/pdf",
            max_bytes=10,
        )


def test_validate_rejects_non_pdf_extension() -> None:
    with pytest.raises(InvalidPDFError, match="\\.pdf"):
        validate_pdf_upload(
            b"%PDF-1.4 fake",
            filename="notes.txt",
            content_type="application/pdf",
            max_bytes=1000,
        )


def test_validate_rejects_bad_magic() -> None:
    with pytest.raises(InvalidPDFError, match="valid PDF"):
        validate_pdf_upload(
            b"not a pdf at all",
            filename="x.pdf",
            content_type="application/pdf",
            max_bytes=1000,
        )


def test_validate_rejects_wrong_content_type() -> None:
    with pytest.raises(InvalidPDFError, match="content type"):
        validate_pdf_upload(
            b"%PDF-1.4",
            filename="x.pdf",
            content_type="text/plain",
            max_bytes=1000,
        )


def test_extract_preserves_page_numbers(sample_pdf_bytes: bytes) -> None:
    parsed = extract_pages(sample_pdf_bytes)
    assert parsed.page_count == 2
    assert [p.page_number for p in parsed.pages] == [1, 2]
    assert "indigo phoenix" in parsed.pages[0].text
    assert "deciduous forests" in parsed.pages[1].text


def test_extract_blank_pdf_raises_ocr_message(blank_pdf_bytes: bytes) -> None:
    with pytest.raises(NoExtractableTextError, match="OCR"):
        extract_pages(blank_pdf_bytes)


def test_extract_corrupt_pdf() -> None:
    from app.exceptions import CorruptPDFError

    with pytest.raises((CorruptPDFError, InvalidPDFError)):
        extract_pages(b"%PDF-1.4\n% definitely corrupt\ntrailer\n")
