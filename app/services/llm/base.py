"""Shared LLM provider contract used by the RAG ask path."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal chat-completion interface for grounded RAG answers."""

    name: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant's final answer text (never thinking/reasoning traces)."""
        ...
