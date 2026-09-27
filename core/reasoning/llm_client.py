"""LLM provider access, isolated behind one interface.

`get_llm_client()` is the only thing the rest of the app should call. It
reads LLM_PROVIDER / GEMINI_API_KEY / GEMINI_MODEL from the environment and
returns whichever client is configured - falling back to a mock client that
needs no API key, so the app runs end-to-end before Gemini is connected.

To connect Gemini later: set GEMINI_API_KEY (and optionally GEMINI_MODEL) in
.env. Nothing else in the app needs to change.
"""
import os
from abc import ABC, abstractmethod
from typing import Optional

# Per-request timeout, and max attempts the SDK will make on a transient 503
# before giving up - configured on the SDK's own retry mechanism (not a
# hand-rolled loop) so there's exactly one retry layer, not two stacked on
# top of each other.
REQUEST_TIMEOUT_MS = 15_000
MAX_ATTEMPTS = 3


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text output for the given prompts."""
        raise NotImplementedError


class QuotaExceededError(Exception):
    """The provider's usage quota is exhausted - unlike a transient overload, retrying
    immediately won't help. Callers should show a distinct message, not "try again"."""


class MockLLMClient(LLMClient):
    """Deterministic stand-in used until a real provider is configured.

    Returns a fixed, validly-shaped JSON response so the rest of the
    pipeline (parsing, display) can be built and tested without any API key.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return """[
  {
    "concept": "[MOCK OUTPUT] Sample creative idea",
    "rationale": "This is placeholder output from MockLLMClient because no LLM_PROVIDER is configured. Set GEMINI_API_KEY in .env to get real generations.",
    "recommended_format": "Instagram Reel",
    "source_context": "Mock client - no real model call was made.",
    "script": null,
    "grounding": "inference",
    "needs_review": false,
    "review_reason": null
  }
]"""


class GeminiClient(LLMClient):
    """`fallback_model` (optional) gets tried automatically if `model`'s quota is
    exhausted - each free-tier model has its own separate daily quota, so a smaller/
    lighter fallback model can often still answer when the primary one can't."""

    def __init__(self, model: str, fallback_model: Optional[str] = None):
        from google import genai  # imported lazily: only required once Gemini is actually used
        from google.genai import types

        http_options = types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retryOptions=types.HttpRetryOptions(attempts=MAX_ATTEMPTS, httpStatusCodes=[503]),
        )
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=http_options)
        self._model = model
        self._fallback_model = fallback_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import errors

        try:
            return self._generate(self._model, system_prompt, user_prompt)
        except (QuotaExceededError, errors.ServerError):
            # Quota exhausted (429) is a hard limit; a ServerError (503) means the SDK's own
            # retries against this same model were already exhausted - either way, retrying
            # the same model again won't help, but a different model might still be up.
            if not self._fallback_model:
                raise
            return self._generate(self._fallback_model, system_prompt, user_prompt)

    def _generate(self, model: str, system_prompt: str, user_prompt: str) -> str:
        from google.genai import errors

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            return response.text
        except errors.ClientError as e:
            if e.code == 429:
                # Quota exhausted - a hard limit, not a blip. Retrying the same model won't help.
                raise QuotaExceededError(str(e)) from e
            raise


def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return GeminiClient(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest"),
        )

    return MockLLMClient()
