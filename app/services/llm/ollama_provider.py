"""Local Ollama HTTP chat provider (localhost only; no paid fallback)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from app.exceptions import (
    OllamaConnectionError,
    OllamaModelMissingError,
    OllamaProviderError,
    OllamaTimeoutError,
)

logger = logging.getLogger(__name__)

# Defense-in-depth: strip accidental thinking tags from final answer text.
_THINK_BLOCK_RE = re.compile(
    r"<think>[\s\S]*?</think>|<thinking>[\s\S]*?</thinking>",
    re.IGNORECASE,
)

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def assert_ollama_localhost(base_url: str) -> str:
    """Require Ollama URL to target loopback only (never expose publicly)."""
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        raise OllamaProviderError(
            "OLLAMA_BASE_URL is empty",
            status_code=503,
        )
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = (parsed.hostname or "").lower()
    if host not in _LOCAL_HOSTS:
        raise OllamaProviderError(
            "OLLAMA_BASE_URL must point to localhost "
            f"(127.0.0.1 / localhost / ::1); got host={host!r}. "
            "Refusing non-local Ollama endpoints.",
            status_code=503,
        )
    scheme = (parsed.scheme or "http").lower()
    if scheme not in {"http", "https"}:
        raise OllamaProviderError(
            f"Unsupported OLLAMA_BASE_URL scheme: {scheme}",
            status_code=503,
        )
    netloc = parsed.netloc or host
    return f"{scheme}://{netloc}".rstrip("/")


def strip_thinking_text(text: str) -> str:
    """Remove thinking/reasoning blocks; return final answer only."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    return cleaned.strip()


def extract_final_answer(message: dict[str, Any]) -> str:
    """Prefer ``content``; never return thinking-only traces to callers/UI."""
    content = message.get("content")
    thinking = message.get("thinking")
    content_s = strip_thinking_text(str(content) if content is not None else "")
    if content_s:
        return content_s
    # Empty content with thinking present = model returned reasoning only.
    # Do not expose thinking; treat as empty so caller can error clearly.
    if thinking is not None and str(thinking).strip():
        logger.warning(
            "Ollama returned thinking without final content; discarding thinking"
        )
    return ""


class OllamaProvider:
    """Chat via Ollama ``/api/chat`` on the local machine."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        num_predict: int = 512,
        think: bool = False,
        client: httpx.Client | None = None,
        check_availability: bool = True,
    ) -> None:
        self.base_url = assert_ollama_localhost(base_url)
        self.model = (model or "").strip()
        if not self.model:
            raise OllamaModelMissingError("OLLAMA_MODEL is empty")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self.num_predict = max(64, int(num_predict))
        self.think = bool(think)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_seconds),
        )
        if check_availability:
            self.ensure_available()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def ensure_available(self) -> None:
        """Verify Ollama is reachable and the configured model is present."""
        try:
            response = self._client.get("/api/tags")
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                "Timed out checking Ollama availability at "
                f"{self.base_url}. Is `ollama serve` running locally?"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                "Could not reach Ollama at "
                f"{self.base_url}. Start it with `ollama serve` "
                "(localhost only; no OpenAI fallback)."
            ) from exc

        if response.status_code >= 400:
            raise OllamaConnectionError(
                f"Ollama /api/tags returned HTTP {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaProviderError(
                "Ollama /api/tags returned non-JSON"
            ) from exc

        names = {
            str(m.get("name") or m.get("model") or "").strip()
            for m in (payload.get("models") or [])
            if isinstance(m, dict)
        }
        names.discard("")
        if self.model not in names:
            available = ", ".join(sorted(names)) or "(none)"
            raise OllamaModelMissingError(
                f"Ollama model {self.model!r} is not available locally. "
                f"Pull it with `ollama pull {self.model}`. "
                f"Installed: {available}. "
                "No OpenAI fallback will be used."
            )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Disable Qwen3 / thinking models' reasoning channel when supported.
            "think": self.think,
            "options": {
                "temperature": temperature,
                "num_predict": self.num_predict,
            },
        }

        last_error: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._client.post("/api/chat", json=body)
            except httpx.TimeoutException as exc:
                last_error = OllamaTimeoutError(
                    f"Ollama chat timed out after {self.timeout_seconds:.0f}s"
                )
                logger.warning(
                    "Ollama timeout attempt=%s/%s", attempt + 1, attempts
                )
            except httpx.HTTPError as exc:
                last_error = OllamaConnectionError(
                    f"Lost connection to Ollama at {self.base_url}"
                )
                logger.warning(
                    "Ollama connection error attempt=%s/%s: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
            else:
                if response.status_code == 404:
                    # Model missing mid-flight
                    detail = _safe_error_detail(response)
                    raise OllamaModelMissingError(
                        detail
                        or f"Ollama model {self.model!r} not found (HTTP 404). "
                        "No OpenAI fallback."
                    )
                if response.status_code >= 500:
                    last_error = OllamaProviderError(
                        f"Ollama server error (HTTP {response.status_code})",
                        status_code=502,
                    )
                    logger.warning(
                        "Ollama 5xx attempt=%s/%s status=%s",
                        attempt + 1,
                        attempts,
                        response.status_code,
                    )
                elif response.status_code >= 400:
                    detail = _safe_error_detail(response)
                    raise OllamaProviderError(
                        detail
                        or f"Ollama request failed (HTTP {response.status_code})",
                        status_code=502,
                    )
                else:
                    return self._parse_chat_response(response)

            if attempt + 1 < attempts:
                time.sleep(min(0.5 * (2**attempt), 2.0))

        assert last_error is not None
        raise last_error

    def _parse_chat_response(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaProviderError(
                "Ollama returned non-JSON chat response"
            ) from exc

        message = payload.get("message")
        if not isinstance(message, dict):
            raise OllamaProviderError("Unexpected Ollama chat response shape")

        answer = extract_final_answer(message)
        if not answer:
            raise OllamaProviderError(
                "Ollama returned an empty answer "
                "(thinking-only or blank content was discarded)"
            )
        return answer


def _safe_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return (response.text or "").strip()[:300]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, str) and err.strip():
            return err.strip()
    return ""
