"""Sentence Transformers embeddings (Phase 2 — not implemented yet)."""


class EmbeddingService:
    """Wrapper around a Sentence Transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_documents(self, _texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""
        raise NotImplementedError("Embeddings land in Phase 2")

    def embed_query(self, _text: str) -> list[float]:
        """Embed a single query string."""
        raise NotImplementedError("Embeddings land in Phase 2")
