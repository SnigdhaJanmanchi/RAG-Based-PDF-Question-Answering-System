"""PDF validation and text extraction with PyMuPDF."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pymupdf as fitz

from app.exceptions import (
    CorruptPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    OversizedPDFError,
    PageExtractionError,
)

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single PDF page (1-based page_number)."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedPDF:
    """Result of parsing a PDF document."""

    pages: list[PageText]
    page_count: int
    metadata: dict[str, str]


def validate_pdf_upload(
    data: bytes,
    *,
    filename: str | None,
    content_type: str | None,
    max_bytes: int,
) -> None:
    """Validate size, content type / extension, and PDF magic bytes."""
    if not data:
        raise InvalidPDFError("Uploaded file is empty")

    if len(data) > max_bytes:
        raise OversizedPDFError(
            f"PDF exceeds maximum upload size of {max_bytes} bytes "
            f"({len(data)} bytes received)"
        )

    name = (filename or "").lower()
    ctype = (content_type or "").lower().split(";")[0].strip()
    allowed_types = {"application/pdf", "application/x-pdf", "binary/octet-stream", ""}
    if ctype and ctype not in allowed_types and ctype != "application/octet-stream":
        # Reject clear non-PDF MIME types (e.g. text/plain, image/png).
        if "pdf" not in ctype:
            raise InvalidPDFError(
                f"Only PDF uploads are accepted (got content type '{content_type}')"
            )

    if name and not name.endswith(".pdf"):
        raise InvalidPDFError("Only files with a .pdf extension are accepted")

    if not data.lstrip().startswith(PDF_MAGIC) and not data.startswith(PDF_MAGIC):
        # Allow optional leading whitespace/BOM-less; require %PDF near start.
        head = data[:1024]
        if PDF_MAGIC not in head:
            raise InvalidPDFError("File content is not a valid PDF (missing %PDF header)")


def extract_pages(data: bytes) -> ParsedPDF:
    """Extract page-wise text with PyMuPDF.

    Empty pages (no text) are retained in the page list with empty strings so
    callers can inspect coverage. Extraction *failures* raise PageExtractionError
    and are never silently skipped.
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — surface as corrupt PDF
        raise CorruptPDFError(f"Corrupt or unreadable PDF: {exc}") from exc

    try:
        if doc.is_encrypted:
            # Try empty password; if still locked, reject.
            if not doc.authenticate(""):
                raise EncryptedPDFError()

        if doc.page_count == 0:
            raise InvalidPDFError("PDF has no pages")

        meta_raw = doc.metadata or {}
        metadata = {
            str(k): str(v)
            for k, v in meta_raw.items()
            if v is not None and str(v).strip()
        }

        pages: list[PageText] = []
        for index in range(doc.page_count):
            page_number = index + 1
            try:
                page = doc.load_page(index)
                text = page.get_text("text") or ""
            except Exception as exc:  # noqa: BLE001
                raise PageExtractionError(page_number, str(exc)) from exc
            pages.append(PageText(page_number=page_number, text=text))

        if not any(p.text.strip() for p in pages):
            raise NoExtractableTextError()

        logger.info(
            "Extracted text from %s/%s pages",
            sum(1 for p in pages if p.text.strip()),
            len(pages),
        )
        return ParsedPDF(pages=pages, page_count=doc.page_count, metadata=metadata)
    finally:
        doc.close()
