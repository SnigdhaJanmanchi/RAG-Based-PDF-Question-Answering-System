"""Recursive text chunking with page-traceable metadata."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.pdf_parser import PageText


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text with provenance metadata."""

    doc_id: str
    chunk_id: str
    chunk_index: int
    page: int
    text: str
    filename: str


def chunk_pages(
    pages: list[PageText],
    *,
    doc_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Split non-empty pages into overlapping chunks.

    Each page is chunked independently so every chunk maps to a single page.
    Pages with no extractable text are omitted from the chunk list (they were
    already validated at parse time as not the sole content of the document).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[TextChunk] = []
    chunk_index = 0
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        pieces = splitter.split_text(text)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{doc_id}:p{page.page_number}:c{chunk_index}"
            chunks.append(
                TextChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    page=page.page_number,
                    text=piece,
                    filename=filename,
                )
            )
            chunk_index += 1

    if not chunks:
        raise ValueError("No chunks produced from extracted pages")

    return chunks
