"""Unit tests for Ollama provider (mocked HTTP) and provider switching."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.exceptions import (
    InvalidLLMProviderError,
    MissingOpenAIKeyError,
    OllamaConnectionError,
    OllamaModelMissingError,
    OllamaProviderError,
    OllamaTimeoutError,
)
from app.services.llm.factory import build_llm_provider
from app.services.llm.ollama_provider import (
    OllamaProvider,
    assert_ollama_localhost,
    extract_final_answer,
    strip_thinking_text,
)
from app.services.llm.openai_provider import OpenAIProvider


def _tags_payload(*models: str) -> dict[str, Any]:
    return {"models": [{"name": m, "model": m} for m in models]}


def _chat_payload(content: str, *, thinking: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {"message": message, "done": True}


def _mock_transport(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    )


def test_assert_ollama_localhost_accepts_loopback() -> None:
    assert assert_ollama_localhost("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert assert_ollama_localhost("http://localhost:11434/") == "http://localhost:11434"


def test_assert_ollama_localhost_rejects_remote() -> None:
    with pytest.raises(OllamaProviderError, match="localhost"):
        assert_ollama_localhost("http://example.com:11434")


def test_strip_thinking_and_extract_final() -> None:
    assert strip_thinking_text("<think>secret</think>Final") == "Final"
    assert (
        extract_final_answer({"content": "Answer", "thinking": "hidden"}) == "Answer"
    )
    # Thinking-only must not leak
    assert extract_final_answer({"content": "", "thinking": "do not leak"}) == ""
    assert extract_final_answer({"content": "<think>x</think>\nYep"}) == "Yep"


def test_settings_default_provider_is_ollama() -> None:
    # Field default (local-dev) is ollama; explicit validate confirms related defaults.
    field_default = Settings.model_fields["llm_provider"].default
    assert field_default == "ollama"
    s = Settings.model_validate(
        {
            "LLM_PROVIDER": "ollama",
            "OPENAI_API_KEY": "",
        }
    )
    assert s.llm_provider == "ollama"
    assert s.ollama_model == "qwen3:0.6b"
    assert s.ollama_base_url == "http://127.0.0.1:11434"


def test_settings_rejects_invalid_provider() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"LLM_PROVIDER": "anthropic"})


def test_factory_openai_requires_key() -> None:
    settings = Settings.model_validate(
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""}
    )
    with pytest.raises(MissingOpenAIKeyError):
        build_llm_provider(settings)


def test_factory_invalid_provider_runtime() -> None:
    settings = Settings.model_validate({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "x"})
    # Bypass Literal by mutating for the factory path edge case
    object.__setattr__(settings, "llm_provider", "bogus")  # type: ignore[arg-type]
    with pytest.raises(InvalidLLMProviderError):
        build_llm_provider(settings)


def test_factory_returns_openai_provider() -> None:
    settings = Settings.model_validate(
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}
    )
    provider = build_llm_provider(settings, openai_client=object())
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"


def test_ollama_complete_mocked_http() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_tags_payload("qwen3:0.6b"))
        assert request.url.path == "/api/chat"
        body = json.loads(request.content.decode())
        seen["body"] = body
        return httpx.Response(
            200,
            json=_chat_payload("Lake Azure, page 1."),
        )

    client = _mock_transport(handler)
    provider = OllamaProvider(
        base_url="http://127.0.0.1:11434",
        model="qwen3:0.6b",
        client=client,
        check_availability=True,
        num_predict=256,
        think=False,
    )
    answer = provider.complete(
        [{"role": "user", "content": "Where?"}],
        temperature=0.0,
    )
    assert answer == "Lake Azure, page 1."
    assert seen["body"]["think"] is False
    assert seen["body"]["options"]["num_predict"] == 256
    assert seen["body"]["stream"] is False
    client.close()


def test_ollama_discards_thinking_only_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_tags_payload("qwen3:0.6b"))
        return httpx.Response(
            200,
            json=_chat_payload("", thinking="I am thinking hard"),
        )

    client = _mock_transport(handler)
    provider = OllamaProvider(
        base_url="http://127.0.0.1:11434",
        model="qwen3:0.6b",
        client=client,
        check_availability=True,
    )
    with pytest.raises(OllamaProviderError, match="empty answer"):
        provider.complete([{"role": "user", "content": "Hi"}])
    client.close()


def test_ollama_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tags_payload("other:1b"))

    client = _mock_transport(handler)
    with pytest.raises(OllamaModelMissingError, match="qwen3:0.6b"):
        OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="qwen3:0.6b",
            client=client,
            check_availability=True,
        )
    client.close()


def test_ollama_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _mock_transport(handler)
    with pytest.raises(OllamaConnectionError):
        OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="qwen3:0.6b",
            client=client,
            check_availability=True,
        )
    client.close()


def test_ollama_retries_then_timeout() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_tags_payload("qwen3:0.6b"))
        calls["n"] += 1
        raise httpx.ReadTimeout("slow", request=request)

    client = _mock_transport(handler)
    provider = OllamaProvider(
        base_url="http://127.0.0.1:11434",
        model="qwen3:0.6b",
        client=client,
        check_availability=True,
        max_retries=2,
        timeout_seconds=1.0,
    )
    with pytest.raises(OllamaTimeoutError):
        provider.complete([{"role": "user", "content": "Hi"}])
    assert calls["n"] == 3  # initial + 2 retries
    client.close()


def test_ollama_no_openai_fallback_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider failures must not silently call OpenAI."""
    called = {"openai": False}

    def boom_openai(*_a: Any, **_k: Any) -> str:
        called["openai"] = True
        return "should never run"

    monkeypatch.setattr(
        "app.services.llm.openai_provider.chat_completion", boom_openai
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_tags_payload("qwen3:0.6b"))
        return httpx.Response(500, json={"error": "boom"})

    client = _mock_transport(handler)
    settings = Settings.model_validate(
        {
            "LLM_PROVIDER": "ollama",
            "OPENAI_API_KEY": "sk-would-be-used-if-fallback",
            "OLLAMA_MODEL": "qwen3:0.6b",
        }
    )
    provider = build_llm_provider(
        settings, ollama_client=client, check_ollama=True
    )
    with pytest.raises(OllamaProviderError):
        provider.complete([{"role": "user", "content": "Hi"}])
    assert called["openai"] is False
    client.close()


def test_ask_api_ollama_mocked_without_openai_key(
    client, sample_pdf_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.15")
    get_settings.cache_clear()

    class FakeOllama:
        name = "ollama"

        def complete(self, messages: list, *, temperature: float = 0.0) -> str:
            assert "UNTRUSTED_PDF_EXCERPTS_BEGIN" in messages[1]["content"]
            return "The indigo phoenix nests at Lake Azure."

    monkeypatch.setattr(
        "app.services.rag.build_llm_provider",
        lambda settings, **kw: FakeOllama(),
    )

    upload = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert upload.status_code == 200, upload.text
    response = client.post(
        "/ask",
        json={"question": "Where does the indigo phoenix nest?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert "Lake Azure" in body["answer"]
    assert body["sources"]
    get_settings.cache_clear()


def test_provider_switch_env_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_provider == "openai"
    p = build_llm_provider(s, openai_client=object())
    assert p.name == "openai"

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    get_settings.cache_clear()
    s2 = get_settings()
    assert s2.llm_provider == "ollama"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tags_payload(s2.ollama_model))

    client = _mock_transport(handler)
    p2 = build_llm_provider(s2, ollama_client=client, check_ollama=True)
    assert p2.name == "ollama"
    client.close()
    get_settings.cache_clear()
