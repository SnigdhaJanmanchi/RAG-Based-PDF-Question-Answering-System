"""Build the configured LLM provider (env switch only; no silent fallback)."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.exceptions import InvalidLLMProviderError, MissingOpenAIKeyError
from app.services.llm.base import LLMProvider
from app.services.llm.ollama_provider import OllamaProvider
from app.services.llm.openai_provider import OpenAIProvider


def build_llm_provider(
    settings: Settings,
    *,
    openai_client: Any | None = None,
    ollama_client: Any | None = None,
    check_ollama: bool = True,
) -> LLMProvider:
    """Construct the active provider from ``settings.llm_provider``.

    Never falls back from Ollama → OpenAI (or the reverse). Switching is
    env-only via ``LLM_PROVIDER``.
    """
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
            max_retries=settings.ollama_max_retries,
            num_predict=settings.ollama_num_predict,
            think=False,
            client=ollama_client,
            check_availability=check_ollama,
        )
    if provider == "openai":
        key = (settings.openai_api_key or "").strip()
        if openai_client is None and not key:
            raise MissingOpenAIKeyError()
        return OpenAIProvider(
            api_key=key,
            model=settings.openai_model,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            client=openai_client,
        )
    raise InvalidLLMProviderError(provider)
