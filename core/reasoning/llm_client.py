"""LLM provider access, isolated behind one interface.

`get_llm_client()` is the only thing the rest of the app should call. It builds an ordered
chain of free-tier model attempts from whatever's configured in the environment - Gemini
models (GEMINI_API_KEY / GEMINI_MODELS), optionally Groq (GROQ_API_KEY / GROQ_MODEL), and
optionally a local Ollama server (OLLAMA_MODEL) - and walks the chain on each request,
advancing past any model that's quota-exhausted, overloaded, times out, or otherwise can't
serve the request, stopping at the first success. If nothing is configured at all, it falls
back to MockLLMClient so the app still runs end-to-end without any credentials.

This project intentionally never uses a paid key: every provider here has a genuine free
tier, and the whole point of the chain is to get reliability *from* that constraint (many
free models tried in order) rather than working around it with a paid one.
"""
import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Per-request timeout. Deliberately no `retryOptions` is set on Gemini's http_options below -
# leaving it unset makes the SDK stop after exactly one attempt per call (see
# google/genai/_api_client.py:retry_args - "If None, the 'never retry' stop strategy will be
# used"), so a model failing is handled once, explicitly, by advancing the chain to the next
# model - not doubled up with the SDK's own internal same-model retry loop.
REQUEST_TIMEOUT_MS = 60_000

# The current primary, then gemini-flash-lite-latest (both requested explicitly), then other
# free-tier flash models - confirmed to actually exist via a live `client.models.list()` call
# against this project's own key, not guessed. gemini-2.5-flash and gemini-2.5-flash-lite were
# deliberately left out: a live call confirmed both now 404 ("no longer available to new
# users") despite still being listed.
DEFAULT_GEMINI_MODELS = "gemini-3.6-flash,gemini-flash-lite-latest,gemini-flash-latest,gemini-3.5-flash,gemini-3.7-flash,gemini-3.8-flash"

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

CACHE_DIR = Path("data/llm_cache")

# Gemini's structured-output schema (an OpenAPI 3.0 subset, not full JSON Schema - hence
# "nullable": true instead of a type union) mirroring core.models.CreativeIdea's fields
# exactly, so response_mime_type="application/json" + this schema make the model's output
# already match what _parse_ideas expects, rather than relying on _parse_ideas to fix it up.
CREATIVE_IDEA_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "concept": {"type": "string"},
            "rationale": {"type": "string"},
            "recommended_format": {"type": "string"},
            "source_context": {"type": "string"},
            "script": {"type": "string", "nullable": True},
            "grounding": {"type": "string", "enum": ["brand_data", "inference"]},
            "needs_review": {"type": "boolean"},
            "review_reason": {"type": "string", "nullable": True},
        },
        "required": ["concept", "rationale", "recommended_format", "source_context", "grounding", "needs_review"],
    },
}


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[dict] = None) -> str:
        """Return raw text output for the given prompts.

        `response_schema` (an OpenAPI-3.0-subset dict, e.g. CREATIVE_IDEA_SCHEMA below)
        constrains structured output on providers that support it (currently just Gemini,
        via response_mime_type="application/json"). Providers that don't support it ignore
        it - callers still get best-effort JSON via the prompt's own instructions, parsed
        leniently, exactly as before this parameter existed. Left as None, Gemini still asks
        for valid JSON but doesn't constrain its shape.
        """
        raise NotImplementedError


class RecoverableProviderError(Exception):
    """Base for one model/provider failing in a way that means 'try the next one in the
    chain', not 'the whole request failed'."""


class QuotaExceededError(RecoverableProviderError):
    """This model's usage quota is exhausted - a hard daily limit for it specifically, not a
    blip, but each free-tier model has its own separate quota so another one may still work."""


class ProviderOverloadedError(RecoverableProviderError):
    """The provider returned a transient server-side overload (e.g. Gemini's 503)."""


class ProviderTimeoutError(RecoverableProviderError):
    """The request to this provider timed out."""


class ProviderRequestError(RecoverableProviderError):
    """This model rejected the request outright (e.g. a 4xx other than quota).

    Confirmed empirically: some free-tier Gemini models (gemini-flash-lite-latest,
    gemini-3.5-flash-lite) return 400 INVALID_ARGUMENT when thinking_config.thinking_budget=0
    is set, while sibling flash models accept it fine - there's no metadata flag to predict
    this per model. Treating it as chain-advance rather than a hard failure means one
    incompatible model (including gemini-flash-lite-latest, which is required in the default
    chain) can't take down the whole request when other models in the chain would work.
    """


class ChainExhaustedError(Exception):
    """Every model in the chain failed. Carries what each one actually returned so the
    failure is debuggable - not "try again", which implies retrying would plausibly help."""

    def __init__(self, attempts: list[tuple[str, Exception]]):
        self.attempts = attempts
        detail = "; ".join(f"{label} -> {type(exc).__name__}: {exc}" for label, exc in attempts)
        super().__init__(f"All {len(attempts)} model(s) in the chain failed: {detail}")


class MockLLMClient(LLMClient):
    """Deterministic stand-in used until a real provider is configured.

    Returns a fixed, validly-shaped JSON response so the rest of the
    pipeline (parsing, display) can be built and tested without any API key.
    """

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[dict] = None) -> str:
        return """[
  {
    "concept": "[MOCK OUTPUT] Sample creative idea",
    "rationale": "This is placeholder output from MockLLMClient because no LLM provider is configured. Set GEMINI_API_KEY (or GROQ_API_KEY / OLLAMA_MODEL) in .env to get real generations.",
    "recommended_format": "Instagram Reel",
    "source_context": "Mock client - no real model call was made.",
    "script": null,
    "grounding": "inference",
    "needs_review": false,
    "review_reason": null
  }
]"""


class GeminiClient(LLMClient):
    """One Gemini model, one attempt per call - chain-level retry/failover across models
    lives in ModelChainClient, not here, so there's exactly one place that decides "try the
    next model" instead of two layers disagreeing with each other."""

    def __init__(self, genai_client, model: str):
        self._client = genai_client
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[dict] = None) -> str:
        import httpx
        from google.genai import errors, types

        config_kwargs = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        config = types.GenerateContentConfig(**config_kwargs)
        try:
            response = self._client.models.generate_content(model=self._model, contents=user_prompt, config=config)
            return response.text
        except errors.ClientError as e:
            if e.code == 429:
                raise QuotaExceededError(str(e)) from e
            raise ProviderRequestError(f"{e.code}: {e}") from e
        except errors.ServerError as e:
            raise ProviderOverloadedError(str(e)) from e
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(str(e)) from e


class GroqClient(LLMClient):
    """Talks to Groq's OpenAI-compatible chat completions endpoint directly over HTTP (via
    httpx, already a dependency of google-genai) rather than adding a whole new SDK for one
    endpoint. A second, independent free provider so a request can still succeed even if
    every Gemini model is unavailable at once."""

    _ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[dict] = None) -> str:
        import httpx

        try:
            response = httpx.post(
                self._ENDPOINT,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=REQUEST_TIMEOUT_MS / 1000,
            )
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(str(e)) from e
        except httpx.HTTPError as e:
            raise ProviderOverloadedError(str(e)) from e

        if response.status_code == 429:
            raise QuotaExceededError(response.text)
        if response.status_code >= 500:
            raise ProviderOverloadedError(response.text)
        if response.status_code >= 400:
            raise ProviderRequestError(f"{response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]


class OllamaClient(LLMClient):
    """Talks to a locally running Ollama server - zero API key, zero network dependency, the
    final fallback for when neither Gemini's nor Groq's free tiers are reachable at all. Only
    ever useful on a machine that actually has Ollama running (never true on a hosted
    deployment like Streamlit Community Cloud) - see is_reachable(), checked once at chain-
    construction time in get_llm_client() rather than on every request."""

    _BASE_URL = "http://localhost:11434"
    _ENDPOINT = f"{_BASE_URL}/api/chat"

    def __init__(self, model: str):
        self._model = model

    @staticmethod
    def is_reachable() -> bool:
        import httpx

        try:
            httpx.get(OllamaClient._BASE_URL, timeout=1.0)
            return True
        except httpx.HTTPError:
            return False

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[dict] = None) -> str:
        import httpx

        try:
            response = httpx.post(
                self._ENDPOINT,
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                timeout=REQUEST_TIMEOUT_MS / 1000,
            )
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(str(e)) from e
        except httpx.HTTPError as e:
            # Most commonly: connection refused because no local Ollama server is running.
            raise ProviderOverloadedError(str(e)) from e

        if response.status_code >= 500:
            raise ProviderOverloadedError(response.text)
        if response.status_code >= 400:
            raise ProviderRequestError(f"{response.status_code}: {response.text}")

        return response.json()["message"]["content"]


class ResponseCache:
    """Dev-only disk cache keyed by a hash of (system_prompt, user_prompt), enabled with
    LLM_CACHE=1. Lets repeated identical prompts during development skip the network (and
    quota) entirely - never used unless explicitly opted into, and never a substitute for
    correctness testing."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, system_prompt: str, user_prompt: str) -> Path:
        digest = hashlib.sha256(f"{system_prompt}\n---\n{user_prompt}".encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.json"

    def get(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        path = self._path(system_prompt, user_prompt)
        if not path.exists():
            return None
        return json.loads(path.read_text())["response"]

    def set(self, system_prompt: str, user_prompt: str, response: str) -> None:
        self._path(system_prompt, user_prompt).write_text(json.dumps({"response": response}))


class ModelChainClient(LLMClient):
    """Walks an ordered list of (label, client) pairs, advancing to the next on a
    RecoverableProviderError, stopping at the first success. Raises ChainExhaustedError,
    listing every attempt, only if none of them work."""

    def __init__(self, chain: list[tuple[str, LLMClient]], cache: Optional[ResponseCache] = None):
        if not chain:
            raise ValueError("Model chain must have at least one entry.")
        self._chain = chain
        self._cache = cache

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Optional[dict] = None) -> str:
        if self._cache is not None:
            cached = self._cache.get(system_prompt, user_prompt)
            if cached is not None:
                return cached

        attempts: list[tuple[str, Exception]] = []
        for label, client in self._chain:
            try:
                result = client.generate(system_prompt, user_prompt, response_schema=response_schema)
            except RecoverableProviderError as e:
                attempts.append((label, e))
                continue

            if self._cache is not None:
                self._cache.set(system_prompt, user_prompt, result)
            return result

        raise ChainExhaustedError(attempts)


def _parse_model_list(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def get_llm_client() -> LLMClient:
    chain: list[tuple[str, LLMClient]] = []

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        from google import genai
        from google.genai import types

        genai_client = genai.Client(api_key=gemini_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
        for model in _parse_model_list(os.getenv("GEMINI_MODELS", DEFAULT_GEMINI_MODELS)):
            chain.append((f"gemini:{model}", GeminiClient(genai_client, model)))

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        chain.append((f"groq:{groq_model}", GroqClient(groq_key, groq_model)))

    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model and OllamaClient.is_reachable():
        chain.append((f"ollama:{ollama_model}", OllamaClient(ollama_model)))

    if not chain:
        return MockLLMClient()

    cache = ResponseCache() if os.getenv("LLM_CACHE") == "1" else None
    return ModelChainClient(chain, cache=cache)
