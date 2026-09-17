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


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text output for the given prompts."""
        raise NotImplementedError


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
    "script": null
  }
]"""


class GeminiClient(LLMClient):
    def __init__(self, model: str):
        import google.generativeai as genai  # imported lazily: only required once Gemini is actually used

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._model = genai.GenerativeModel(model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._model.generate_content(f"{system_prompt}\n\n{user_prompt}")
        return response.text


def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return GeminiClient(model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    return MockLLMClient()
