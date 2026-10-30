"""FAISS vector index with on-disk persistence and chunk metadata."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """Metadata stored alongside each FAISS vector."""

    vector_id: int
    doc_id: str
    chunk_id: str
    chunk_index: int
    page: int
    text: str
    filename: str


@dataclass
class SearchHit:
    """One similarity search result."""

    score: float
    text: str
    doc_id: str
    chunk_id: str
    chunk_index: int
    page: int
    filename: str


@dataclass
class DocumentRecord:
    """Registry entry for an indexed document."""

    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    content_hash: str
    pdf_metadata: dict[str, str]


class VectorStore:
    """Persist and search document chunk embeddings with FAISS IndexFlatIP."""

    def __init__(self, data_dir: str | Path, dimension: int) -> None:
        self.data_dir = Path(data_dir)
        self.dimension = dimension
        self._lock = threading.RLock()
        self._index_dir = self.data_dir / "faiss"
        self._uploads_dir = self.data_dir / "uploads"
        self._index_path = self._index_dir / "index.faiss"
        self._chunks_path = self._index_dir / "chunks.json"
        self._docs_path = self._index_dir / "documents.json"
        self._next_id = 0
        self._chunks: dict[int, ChunkRecord] = {}
        self._documents: dict[str, DocumentRecord] = {}
        self._index: faiss.IndexIDMap2 | None = None
        self._ensure_dirs()
        self._load_or_create()

    def _ensure_dirs(self) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

    def _create_empty_index(self) -> faiss.IndexIDMap2:
        base = faiss.IndexFlatIP(self.dimension)
        return faiss.IndexIDMap2(base)

    def _load_or_create(self) -> None:
        with self._lock:
            if self._index_path.exists() and self._chunks_path.exists():
                try:
                    self._index = faiss.read_index(str(self._index_path))
                    raw_chunks = json.loads(self._chunks_path.read_text(encoding="utf-8"))
                    self._chunks = {
                        int(item["vector_id"]): ChunkRecord(**item) for item in raw_chunks
                    }
                    if self._docs_path.exists():
                        raw_docs = json.loads(self._docs_path.read_text(encoding="utf-8"))
                        self._documents = {
                            item["doc_id"]: DocumentRecord(**item) for item in raw_docs
                        }
                    if self._chunks:
                        self._next_id = max(self._chunks.keys()) + 1
                    # Dimension check
                    if self._index.d != self.dimension:
                        raise ValueError(
                            f"Persisted FAISS dim {self._index.d} != expected {self.dimension}"
                        )
                    logger.info(
                        "Loaded FAISS index (%s vectors, %s documents)",
                        self._index.ntotal,
                        len(self._documents),
                    )
                    return
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to load FAISS store; recreating empty index")

            self._index = self._create_empty_index()
            self._chunks = {}
            self._documents = {}
            self._next_id = 0
            self._persist()

    def _persist(self) -> None:
        assert self._index is not None
        # Write to temp files then replace for safer persistence.
        tmp_index = self._index_path.with_suffix(".faiss.tmp")
        tmp_chunks = self._chunks_path.with_suffix(".json.tmp")
        tmp_docs = self._docs_path.with_suffix(".json.tmp")

        faiss.write_index(self._index, str(tmp_index))
        tmp_chunks.write_text(
            json.dumps([asdict(c) for c in self._chunks.values()], indent=2),
            encoding="utf-8",
        )
        tmp_docs.write_text(
            json.dumps([asdict(d) for d in self._documents.values()], indent=2),
            encoding="utf-8",
        )
        tmp_index.replace(self._index_path)
        tmp_chunks.replace(self._chunks_path)
        tmp_docs.replace(self._docs_path)

    @property
    def uploads_dir(self) -> Path:
        return self._uploads_dir

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._lock:
            for doc in self._documents.values():
                if doc.content_hash == content_hash:
                    return doc
            return None

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._documents.values(), key=lambda d: d.filename.lower())

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._documents.get(doc_id)

    def _purge_document_locked(self, doc_id: str, *, delete_upload: bool) -> bool:
        """Remove vectors + registry for doc_id. Caller must hold ``_lock``."""
        if doc_id not in self._documents:
            return False
        assert self._index is not None
        ids_to_remove = [vid for vid, c in self._chunks.items() if c.doc_id == doc_id]
        if ids_to_remove:
            id_array = np.asarray(ids_to_remove, dtype=np.int64)
            self._index.remove_ids(id_array)
        self._chunks = {
            vid: c for vid, c in self._chunks.items() if c.doc_id != doc_id
        }
        del self._documents[doc_id]
        if delete_upload:
            upload = self._uploads_dir / f"{doc_id}.pdf"
            if upload.exists():
                upload.unlink()
        return True

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document's vectors, registry entry, and uploaded PDF."""
        with self._lock:
            removed = self._purge_document_locked(doc_id, delete_upload=True)
            if removed:
                self._persist()
            return removed

    def add_documents(
        self,
        *,
        document: DocumentRecord,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Add embedded chunks for a document. Replaces prior vectors if doc_id exists."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings/chunks length mismatch")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dim {embeddings.shape[1]} != store dim {self.dimension}"
            )

        with self._lock:
            assert self._index is not None
            # Replace existing doc_id if re-indexing same id (keep upload file).
            if document.doc_id in self._documents:
                self._purge_document_locked(document.doc_id, delete_upload=False)

            vectors = np.ascontiguousarray(embeddings.astype(np.float32))
            # Ensure L2-normalized for cosine via inner product.
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            vectors = vectors / norms

            ids = np.arange(
                self._next_id, self._next_id + len(chunks), dtype=np.int64
            )
            self._index.add_with_ids(vectors, ids)

            for vid, chunk in zip(ids.tolist(), chunks, strict=True):
                record = ChunkRecord(
                    vector_id=vid,
                    doc_id=document.doc_id,
                    chunk_id=chunk["chunk_id"],
                    chunk_index=chunk["chunk_index"],
                    page=chunk["page"],
                    text=chunk["text"],
                    filename=chunk["filename"],
                )
                self._chunks[vid] = record

            self._next_id = int(ids[-1]) + 1 if len(ids) else self._next_id
            self._documents[document.doc_id] = document
            self._persist()
            logger.info(
                "Indexed document %s (%s chunks); store size=%s",
                document.doc_id,
                len(chunks),
                self._index.ntotal,
            )

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[SearchHit]:
        """Return top-k similar chunks (optionally scoped to one document)."""
        with self._lock:
            assert self._index is not None
            if self._index.ntotal == 0:
                return []

            vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
            if vec.shape[1] != self.dimension:
                raise ValueError(
                    f"query dim {vec.shape[1]} != store dim {self.dimension}"
                )
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm

            # Over-fetch if filtering by doc_id.
            fetch_k = min(self._index.ntotal, max(top_k * 5, top_k) if doc_id else top_k)
            scores, ids = self._index.search(vec, fetch_k)

            hits: list[SearchHit] = []
            for score, vid in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
                if vid < 0:
                    continue
                record = self._chunks.get(int(vid))
                if record is None:
                    continue
                if doc_id is not None and record.doc_id != doc_id:
                    continue
                hits.append(
                    SearchHit(
                        score=float(score),
                        text=record.text,
                        doc_id=record.doc_id,
                        chunk_id=record.chunk_id,
                        chunk_index=record.chunk_index,
                        page=record.page,
                        filename=record.filename,
                    )
                )
                if len(hits) >= top_k:
                    break
            return hits
