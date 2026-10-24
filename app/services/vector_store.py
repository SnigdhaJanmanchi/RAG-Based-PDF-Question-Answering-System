"""FAISS vector index + metadata store (Phase 2 — not implemented yet)."""


class VectorStore:
    """Persist and search document chunk embeddings."""

    def add_documents(self, *_args, **_kwargs) -> None:
        """Add embedded chunks to the index."""
        raise NotImplementedError("Vector store lands in Phase 2")

    def similarity_search(self, *_args, **_kwargs) -> list[dict]:
        """Return top-k similar chunks for a query embedding."""
        raise NotImplementedError("Vector store lands in Phase 2")
