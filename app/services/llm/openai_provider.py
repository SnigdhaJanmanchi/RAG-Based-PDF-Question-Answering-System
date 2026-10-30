"""OpenAI chat provider wrapping the existing SDK helper."""

from __future__ import annotations

from typing import Any

from app.services.openai_client import build_openai_client, chat_completion


class OpenAIProvider:
    """Paid OpenAI Chat Completions provider."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            self._client = build_openai_client(
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        return chat_completion(
            self._client,
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
