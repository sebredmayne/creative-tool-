"""Tests for the free-tier model chain: per-provider error mapping, chain failover/caching,
and how get_llm_client() assembles the chain from environment config. No real API calls."""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.reasoning.llm_client import (
    ChainExhaustedError,
    GeminiClient,
    GroqClient,
    ModelChainClient,
    ProviderOverloadedError,
    ProviderRequestError,
    ProviderTimeoutError,
    QuotaExceededError,
    ResponseCache,
    get_llm_client,
)


# --- GeminiClient: one model, one attempt, mapped to the right recoverable error -----------


@pytest.fixture
def genai_client():
    from google import genai

    return genai.Client(api_key="fake-key-for-tests")


def test_gemini_client_returns_text_on_success(genai_client):
    client = GeminiClient(genai_client, model="some-model")
    with patch.object(genai_client.models, "generate_content", return_value=MagicMock(text="ok")):
        assert client.generate("system", "user") == "ok"


def test_gemini_client_raises_quota_exceeded_on_429(genai_client):
    from google.genai import errors

    client = GeminiClient(genai_client, model="some-model")
    error = errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    with patch.object(genai_client.models, "generate_content", side_effect=error):
        with pytest.raises(QuotaExceededError):
            client.generate("system", "user")


def test_gemini_client_raises_overloaded_on_503(genai_client):
    from google.genai import errors

    client = GeminiClient(genai_client, model="some-model")
    error = errors.ServerError(503, {"error": {"message": "overloaded"}})
    with patch.object(genai_client.models, "generate_content", side_effect=error):
        with pytest.raises(ProviderOverloadedError):
            client.generate("system", "user")


def test_gemini_client_raises_request_error_on_other_4xx(genai_client):
    """Confirmed empirically: some models reject thinking_budget=0 with a plain 400, not a
    429/503 - this must still be recoverable so one incompatible model in the chain
    (gemini-flash-lite-latest, specifically) can't take down the whole request."""
    from google.genai import errors

    client = GeminiClient(genai_client, model="some-model")
    error = errors.ClientError(400, {"error": {"message": "invalid argument"}})
    with patch.object(genai_client.models, "generate_content", side_effect=error):
        with pytest.raises(ProviderRequestError):
            client.generate("system", "user")


def test_gemini_client_raises_timeout_error_on_httpx_timeout(genai_client):
    client = GeminiClient(genai_client, model="some-model")
    with patch.object(genai_client.models, "generate_content", side_effect=httpx.ReadTimeout("timed out")):
        with pytest.raises(ProviderTimeoutError):
            client.generate("system", "user")


# --- GroqClient: OpenAI-compatible HTTP calls, same error mapping --------------------------


def test_groq_client_returns_content_on_success():
    response = MagicMock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": "groq output"}}]}
    with patch("httpx.post", return_value=response):
        client = GroqClient(api_key="fake", model="fake-model")
        assert client.generate("system", "user") == "groq output"


def test_groq_client_raises_quota_exceeded_on_429():
    response = MagicMock(status_code=429, text="rate limited")
    with patch("httpx.post", return_value=response):
        client = GroqClient(api_key="fake", model="fake-model")
        with pytest.raises(QuotaExceededError):
            client.generate("system", "user")


# --- ModelChainClient: failover across models, first-success-wins, cache -------------------


class _FakeClient:
    """Minimal LLMClient stand-in: replays a fixed sequence of results/exceptions."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_schema=None):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.parametrize(
    "error",
    [QuotaExceededError("quota"), ProviderOverloadedError("overloaded"), ProviderTimeoutError("timed out")],
)
def test_chain_advances_to_next_model_on_recoverable_errors(error):
    failing = _FakeClient([error])
    working = _FakeClient(["result"])
    chain = ModelChainClient([("a", failing), ("b", working)])

    assert chain.generate("sys", "user") == "result"
    assert failing.calls == 1
    assert working.calls == 1


def test_chain_stops_at_first_success_without_trying_later_models():
    working = _FakeClient(["first result"])
    never_called = _FakeClient(["should not be used"])
    chain = ModelChainClient([("a", working), ("b", never_called)])

    assert chain.generate("sys", "user") == "first result"
    assert never_called.calls == 0


def test_chain_raises_with_every_attempt_when_all_models_fail():
    a = _FakeClient([QuotaExceededError("a failed")])
    b = _FakeClient([ProviderOverloadedError("b failed")])
    chain = ModelChainClient([("a", a), ("b", b)])

    with pytest.raises(ChainExhaustedError) as exc_info:
        chain.generate("sys", "user")

    assert [label for label, _ in exc_info.value.attempts] == ["a", "b"]


def test_cache_hit_skips_client_call_entirely(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path)
    client = _FakeClient(["real response"])
    chain = ModelChainClient([("a", client)], cache=cache)

    first = chain.generate("sys", "user")
    second = chain.generate("sys", "user")

    assert first == second == "real response"
    assert client.calls == 1  # second call was served from the cache, not the client


# --- get_llm_client(): assembling the chain from environment config -----------------------


def test_get_llm_client_omits_groq_when_key_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    client = get_llm_client()

    assert isinstance(client, ModelChainClient)
    assert not any(label.startswith("groq:") for label, _ in client._chain)


def test_get_llm_client_includes_groq_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    client = get_llm_client()

    assert any(label.startswith("groq:") for label, _ in client._chain)


def test_get_llm_client_falls_back_to_mock_when_nothing_configured(monkeypatch):
    from core.reasoning.llm_client import MockLLMClient

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    assert isinstance(get_llm_client(), MockLLMClient)


def test_get_llm_client_omits_ollama_when_unreachable(monkeypatch):
    from core.reasoning.llm_client import OllamaClient

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setattr(OllamaClient, "is_reachable", staticmethod(lambda: False))

    client = get_llm_client()

    assert not any(label.startswith("ollama:") for label, _ in client._chain)


def test_get_llm_client_includes_ollama_when_reachable(monkeypatch):
    from core.reasoning.llm_client import OllamaClient

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setattr(OllamaClient, "is_reachable", staticmethod(lambda: True))

    client = get_llm_client()

    assert any(label.startswith("ollama:") for label, _ in client._chain)


def test_ollama_is_reachable_does_not_raise_and_returns_quickly():
    """No mocking - a real network check. Whether it's True or False depends on whether this
    machine happens to have Ollama running, but it must complete fast (it's checked at
    chain-construction time) and never raise, which is the actual property worth guarding -
    the whole point is that an absent server (the common case, and always true on a hosted
    deployment) must be handled, not crash chain construction."""
    import time

    from core.reasoning.llm_client import OllamaClient

    start = time.monotonic()
    result = OllamaClient.is_reachable()
    elapsed = time.monotonic() - start

    assert isinstance(result, bool)
    assert elapsed < 5  # generous margin over the 1s timeout inside is_reachable()
