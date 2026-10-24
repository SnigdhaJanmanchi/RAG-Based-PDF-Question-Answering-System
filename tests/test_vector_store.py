"""FAISS persistence and similarity search tests (synthetic embeddings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.vector_store import DocumentRecord, VectorStore


def _doc(doc_id: str, filename: str = "a.pdf", content_hash: str = "h1") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        filename=filename,
        page_count=1,
        chunk_count=2,
        content_hash=content_hash,
        pdf_metadata={},
    )


def test_add_search_and_persist(tmp_path: Path) -> None:
    dim = 8
    store = VectorStore(tmp_path / "data", dimension=dim)

    # Two orthogonal-ish vectors for distinct chunks.
    emb = np.zeros((2, dim), dtype=np.float32)
    emb[0, 0] = 1.0
    emb[1, 1] = 1.0
    chunks = [
        {
            "chunk_id": "d1:p1:c0",
            "chunk_index": 0,
            "page": 1,
            "text": "indigo phoenix nests at Lake Azure",
            "filename": "a.pdf",
        },
        {
            "chunk_id": "d1:p1:c1",
            "chunk_index": 1,
            "page": 1,
            "text": "unrelated deciduous forests",
            "filename": "a.pdf",
        },
    ]
    store.add_documents(document=_doc("d1"), chunks=chunks, embeddings=emb)

    query = np.zeros(dim, dtype=np.float32)
    query[0] = 1.0
    hits = store.similarity_search(query, top_k=1)
    assert len(hits) == 1
    assert "indigo phoenix" in hits[0].text
    assert hits[0].page == 1
    assert hits[0].doc_id == "d1"

    # Reload from disk
    store2 = VectorStore(tmp_path / "data", dimension=dim)
    hits2 = store2.similarity_search(query, top_k=1)
    assert hits2[0].text == hits[0].text
    assert store2.get_document("d1") is not None


def test_duplicate_hash_lookup(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "data", dimension=4)
    emb = np.eye(1, 4, dtype=np.float32)
    chunks = [
        {
            "chunk_id": "d1:p1:c0",
            "chunk_index": 0,
            "page": 1,
            "text": "alpha",
            "filename": "a.pdf",
        }
    ]
    store.add_documents(
        document=_doc("d1", content_hash="abc"),
        chunks=chunks,
        embeddings=emb,
    )
    assert store.get_document_by_hash("abc") is not None
    assert store.get_document_by_hash("zzz") is None


def test_remove_keeps_index_consistent(tmp_path: Path) -> None:
    dim = 4
    store = VectorStore(tmp_path / "data", dimension=dim)
    emb = np.eye(2, dim, dtype=np.float32)
    chunks = [
        {
            "chunk_id": "d1:p1:c0",
            "chunk_index": 0,
            "page": 1,
            "text": "one",
            "filename": "a.pdf",
        },
        {
            "chunk_id": "d1:p1:c1",
            "chunk_index": 1,
            "page": 1,
            "text": "two",
            "filename": "a.pdf",
        },
    ]
    store.add_documents(document=_doc("d1"), chunks=chunks, embeddings=emb)
    assert store.remove_document("d1") is True
    assert store.list_documents() == []
    assert store.similarity_search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=3) == []
