"""Thin OpenAI chat wrapper with timeouts, retries, and typed errors."""

from __future__ import annotations

import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from app.exceptions import (
    MissingOpenAIKeyError,
    OpenAIProviderError,
    OpenAIRateLimitError,
)

logger = logging.getLogger(__name__)


def build_openai_client(
    *,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
) -> OpenAI:
    """Create an official OpenAI SDK client. Raises if the key is missing."""
    key = (api_key or "").strip()
    if not key:
        raise MissingOpenAIKeyError()
    return OpenAI(api_key=key, timeout=timeout_seconds, max_retries=max_retries)


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
) -> str:
    """Call chat.completions and return the assistant text.

    The SDK retries transient failures up to ``max_retries``. Remaining failures
    are mapped to HTTP-friendly domain errors.
    """
    try:
        response: Any = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except AuthenticationError as exc:
        raise MissingOpenAIKeyError(
            "OpenAI authentication failed — check OPENAI_API_KEY"
        ) from exc
    except RateLimitError as exc:
        raise OpenAIRateLimitError() from exc
    except APITimeoutError as exc:
        raise OpenAIProviderError(
            "OpenAI request timed out", status_code=504
        ) from exc
    except APIConnectionError as exc:
        raise OpenAIProviderError(
            "Could not reach OpenAI API", status_code=502
        ) from exc
    except APIStatusError as exc:
        status = int(getattr(exc, "status_code", 502) or 502)
        raise OpenAIProviderError(
            f"OpenAI API error ({status})",
            status_code=502 if status >= 500 else status,
        ) from exc
    except MissingOpenAIKeyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected OpenAI client failure")
        raise OpenAIProviderError(f"OpenAI request failed: {exc}") from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise OpenAIProviderError("Unexpected OpenAI response shape") from exc

    if content is None or not str(content).strip():
        raise OpenAIProviderError("OpenAI returned an empty answer")
    return str(content).strip()
