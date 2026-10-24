"""Retrieve → prompt → OpenAI generate (Phase 3 — not implemented yet)."""


class RAGService:
    """End-to-end question answering over indexed PDFs."""

    def ask(self, *_args, **_kwargs) -> dict:
        """Answer a question using retrieved context and an LLM."""
        raise NotImplementedError("RAG answer path lands in Phase 3")
