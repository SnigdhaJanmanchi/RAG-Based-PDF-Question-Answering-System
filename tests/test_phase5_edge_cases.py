"""Phase 5 edge-case PDFs: corrupt, scanned, tables, unsupported types, path traversal."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.exceptions import CorruptPDFError, InvalidPDFError, NoExtractableTextError
from app.services.ingest import sanitize_filename
from app.services.pdf_parser import extract_pages, validate_pdf_upload

FIXTURES = Path(__file__).parent / "fixtures"


def test_sanitize_filename_blocks_path_traversal() -> None:
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert "/" not in sanitize_filename("foo/bar/baz.pdf")
    assert ".." not in sanitize_filename("../x.pdf")
    assert sanitize_filename("notes.txt").endswith(".pdf")
    assert sanitize_filename(None) == "upload.pdf"


def test_scanned_like_pdf_ocr_message() -> None:
    data = (FIXTURES / "scanned_like.pdf").read_bytes()
    with pytest.raises(NoExtractableTextError, match="OCR"):
        extract_pages(data)


def test_corrupt_pdf_rejected() -> None:
    data = (FIXTURES / "corrupt.pdf").read_bytes()
    with pytest.raises((CorruptPDFError, InvalidPDFError, NoExtractableTextError)):
        extract_pages(data)


def test_unsupported_txt_extension() -> None:
    data = (FIXTURES / "notes.txt").read_bytes()
    with pytest.raises(InvalidPDFError):
        validate_pdf_upload(
            data,
            filename="notes.txt",
            content_type="text/plain",
            max_bytes=1_000_000,
        )


def test_multipage_and_table_text_extractable() -> None:
    parsed = extract_pages((FIXTURES / "multipage_facts.pdf").read_bytes())
    assert parsed.page_count == 3
    assert "WH-NORTH-17" in parsed.pages[0].text
    assert "Blue Courier" in parsed.pages[1].text
    assert "Bin-14" in parsed.pages[2].text
    assert "250 kilograms" in parsed.pages[2].text


def test_api_rejects_scanned_and_corrupt_and_txt(client) -> None:
    scanned = client.post(
        "/documents/upload",
        files={
            "file": (
                "scanned_like.pdf",
                (FIXTURES / "scanned_like.pdf").read_bytes(),
                "application/pdf",
            )
        },
    )
    assert scanned.status_code == 422
    assert "OCR" in scanned.json()["detail"]

    corrupt = client.post(
        "/documents/upload",
        files={
            "file": (
                "corrupt.pdf",
                (FIXTURES / "corrupt.pdf").read_bytes(),
                "application/pdf",
            )
        },
    )
    assert corrupt.status_code in {400, 422}

    txt = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                (FIXTURES / "notes.txt").read_bytes(),
                "text/plain",
            )
        },
    )
    assert txt.status_code == 400
