"""LLM provider abstraction (OpenAI + Ollama)."""

from app.services.llm.base import LLMProvider
from app.services.llm.factory import build_llm_provider

__all__ = ["LLMProvider", "build_llm_provider"]
