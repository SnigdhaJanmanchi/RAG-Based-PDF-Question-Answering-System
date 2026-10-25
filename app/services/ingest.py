"""Orchestrate PDF upload → parse → chunk → embed → FAISS index."""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.services.chunker import TextChunk, chunk_pages
from app.services.embeddings import EmbeddingService
from app.services.pdf_parser import extract_pages, validate_pdf_upload
from app.services.vector_store import DocumentRecord, SearchHit, VectorStore

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str | None, *, fallback: str = "upload.pdf") -> str:
    """Return a basename-only PDF filename (blocks path traversal)."""
    raw = (filename or "").strip() or fallback
    name = Path(raw).name  # drop any directory components
    name = re.sub(r"[^\w.\- ()\[\]]+", "_", name)
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf" if name else fallback
    # Prevent empty / dot-only names
    stem = Path(name).stem.strip(" .")
    if not stem:
        name = fallback
    return name[:180]


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    duplicate: bool


class IngestService:
    """High-level document ingest and search."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        store: VectorStore,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.store = store

    def ingest_pdf(
        self,
        data: bytes,
        *,
        filename: str,
        content_type: str | None,
    ) -> IngestResult:
        validate_pdf_upload(
            data,
            filename=filename,
            content_type=content_type,
            max_bytes=self.settings.max_upload_bytes,
        )

        content_hash = hashlib.sha256(data).hexdigest()
        existing = self.store.get_document_by_hash(content_hash)
        if existing is not None:
            logger.info(
                "Duplicate PDF content (hash=%s) → returning existing doc %s",
                content_hash[:12],
                existing.doc_id,
            )
            return IngestResult(
                document_id=existing.doc_id,
                filename=existing.filename,
                page_count=existing.page_count,
                chunk_count=existing.chunk_count,
                duplicate=True,
            )

        parsed = extract_pages(data)
        doc_id = str(uuid.uuid4())
        safe_name = sanitize_filename(filename, fallback=f"{doc_id}.pdf")

        chunks: list[TextChunk] = chunk_pages(
            parsed.pages,
            doc_id=doc_id,
            filename=safe_name,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        embeddings = self.embeddings.embed_documents([c.text for c in chunks])
        document = DocumentRecord(
            doc_id=doc_id,
            filename=safe_name,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            content_hash=content_hash,
            pdf_metadata=parsed.metadata,
        )
        chunk_payloads = [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "page": c.page,
                "text": c.text,
                "filename": c.filename,
            }
            for c in chunks
        ]

        # Persist original PDF then index (atomic enough for local MVP).
        upload_path = self.store.uploads_dir / f"{doc_id}.pdf"
        upload_path.write_bytes(data)

        try:
            self.store.add_documents(
                document=document,
                chunks=chunk_payloads,
                embeddings=embeddings,
            )
        except Exception:
            if upload_path.exists():
                upload_path.unlink()
            raise

        return IngestResult(
            document_id=doc_id,
            filename=safe_name,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            duplicate=False,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[SearchHit]:
        q = (query or "").strip()
        if not q:
            raise ValueError("query must not be empty")
        k = top_k if top_k is not None else self.settings.top_k
        if k < 1:
            raise ValueError("top_k must be >= 1")
        if doc_id is not None and self.store.get_document(doc_id) is None:
            raise KeyError(f"Unknown document_id: {doc_id}")
        vector = self.embeddings.embed_query(q)
        return self.store.similarity_search(vector, top_k=k, doc_id=doc_id)
