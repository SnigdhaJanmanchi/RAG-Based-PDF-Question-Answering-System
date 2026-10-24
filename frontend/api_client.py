"""HTTP client for the FastAPI backend (Phase 4).

Talks only to existing routes:
  GET  /health
  POST /documents/upload
  GET  /documents
  POST /ask
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 120.0


class APIClientError(Exception):
    """User-safe error from the backend client (no stack traces / secrets)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class DocumentInfo:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True)
class UploadResult:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    duplicate: bool


@dataclass(frozen=True)
class AskSource:
    document_id: str
    filename: str
    page: int
    chunk_id: str
    excerpt: str
    score: float


@dataclass(frozen=True)
class AskResult:
    question: str
    answer: str
    status: str
    sources: list[AskSource]
    documents: list[dict[str, str]]


def resolve_api_base_url(explicit: str | None = None) -> str:
    raw = (explicit or os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL).strip()
    return raw.rstrip("/")


def _user_safe_detail(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            # Never surface stack traces or env secrets if a proxy echoed them.
            text = detail.strip()
            if "Traceback" in text or "OPENAI_API_KEY=" in text:
                return fallback
            return text
        if isinstance(detail, list):
            return fallback
    return fallback


class APIClient:
    """Reusable FastAPI client with connection/request timeouts."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = resolve_api_base_url(base_url)
        timeout = (
            DEFAULT_TIMEOUT_SECONDS
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def list_documents(self) -> list[DocumentInfo]:
        data = self._request_json("GET", "/documents")
        docs = data.get("documents") if isinstance(data, dict) else None
        if not isinstance(docs, list):
            raise APIClientError("Unexpected response from document list")
        result: list[DocumentInfo] = []
        for item in docs:
            if not isinstance(item, dict):
                continue
            result.append(
                DocumentInfo(
                    document_id=str(item["document_id"]),
                    filename=str(item["filename"]),
                    page_count=int(item["page_count"]),
                    chunk_count=int(item["chunk_count"]),
                )
            )
        return result

    def upload_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
    ) -> UploadResult:
        try:
            response = self._client.post(
                "/documents/upload",
                files={"file": (filename, content, content_type)},
            )
        except httpx.ConnectError as exc:
            raise APIClientError(
                "Cannot reach the API. Is FastAPI running "
                f"at {self.base_url}?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise APIClientError(
                "Upload timed out while talking to the API. Try a smaller PDF "
                "or check that the server is still running."
            ) from exc
        except httpx.HTTPError as exc:
            raise APIClientError("Upload failed due to a network error") from exc

        data = self._parse_response(
            response,
            fallback_error="Upload failed",
        )
        return UploadResult(
            document_id=str(data["document_id"]),
            filename=str(data["filename"]),
            page_count=int(data["page_count"]),
            chunk_count=int(data["chunk_count"]),
            duplicate=bool(data.get("duplicate", False)),
        )

    def ask(
        self,
        question: str,
        *,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> AskResult:
        payload: dict[str, Any] = {"question": question}
        if document_id:
            payload["document_id"] = document_id
        if top_k is not None:
            payload["top_k"] = top_k

        data = self._request_json("POST", "/ask", json=payload)
        sources = [
            AskSource(
                document_id=str(s["document_id"]),
                filename=str(s["filename"]),
                page=int(s["page"]),
                chunk_id=str(s["chunk_id"]),
                excerpt=str(s["excerpt"]),
                score=float(s["score"]),
            )
            for s in data.get("sources", [])
            if isinstance(s, dict)
        ]
        documents = [
            {"document_id": str(d["document_id"]), "filename": str(d["filename"])}
            for d in data.get("documents", [])
            if isinstance(d, dict)
        ]
        return AskResult(
            question=str(data.get("question", question)),
            answer=str(data.get("answer", "")),
            status=str(data.get("status", "ok")),
            sources=sources,
            documents=documents,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise APIClientError(
                "Cannot reach the API. Is FastAPI running "
                f"at {self.base_url}?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise APIClientError(
                "The API request timed out. The server may be busy or offline."
            ) from exc
        except httpx.HTTPError as exc:
            raise APIClientError("Network error while contacting the API") from exc

        return self._parse_response(response, fallback_error="API request failed")

    def _parse_response(
        self,
        response: httpx.Response,
        *,
        fallback_error: str,
    ) -> dict[str, Any]:
        status = response.status_code
        try:
            payload: Any = response.json() if response.content else {}
        except ValueError:
            payload = {}

        if status >= 400:
            if status == 503:
                msg = _user_safe_detail(
                    payload,
                    "The answer service is unavailable. Check that OPENAI_API_KEY "
                    "is set on the API server.",
                )
            elif status == 429:
                msg = _user_safe_detail(
                    payload,
                    "The AI provider rate-limited the request. Please try again shortly.",
                )
            elif status == 404:
                msg = _user_safe_detail(payload, "Requested document was not found.")
            elif status >= 500:
                msg = _user_safe_detail(
                    payload,
                    "The API reported an internal error. Please try again.",
                )
            else:
                msg = _user_safe_detail(payload, fallback_error)
            raise APIClientError(msg, status_code=status)

        if not isinstance(payload, dict):
            raise APIClientError("Unexpected response from the API")
        return payload
