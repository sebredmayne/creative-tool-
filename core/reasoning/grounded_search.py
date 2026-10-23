"""Resolves a brand name to its Google Ads Transparency Center advertiser URL, via Gemini's
built-in Google Search grounding tool - the only practical way to find this, since Google
publishes no API for it (see core.google_ads's docstring).

Deliberately NOT an LLMClient implementation and not routed through get_llm_client()'s
provider chain: Gemini's API cannot combine `response_schema` (structured JSON mode) with
`tools` (Search grounding) in the same call, so this always asks for plain text and regex-
extracts the answer, and grounding itself is Gemini-specific - there's no Groq/Ollama
equivalent to fall back to. If the one grounded call doesn't work, there's nothing else to try;
every failure mode here (not configured, request failed, no URL found) collapses to the same
outcome for callers - skip this brand's Google side, same as core.ad_library.is_configured()'s
role for the Meta side.

Caveat for anyone touching this: grounding citation URIs
(response.candidates[0].grounding_metadata.grounding_chunks[].web.uri) are typically
vertexaisearch.cloud.google.com/grounding-api-redirect/... wrapper links, not the real target -
don't bother parsing those. The model's own plain-text answer (format enforced by the prompt
below) is the only signal this relies on.
"""
import os
import re
from typing import Optional

from core.google_ads import ADVERTISER_URL_RE
from core.reasoning.llm_client import DEFAULT_GEMINI_MODELS, REQUEST_TIMEOUT_MS, _parse_model_list

GROUNDING_SYSTEM_PROMPT = (
    "You have access to Google Search. Find this brand's advertiser page on Google's Ads "
    "Transparency Center (adstransparency.google.com). That page's URL has the exact form "
    "https://adstransparency.google.com/advertiser/AR<id>/... - search things like "
    '"<brand> site:adstransparency.google.com" or "<brand> google ads transparency center '
    'advertiser" and look for that exact URL pattern in search results or page content.\n\n'
    "Respond with plain text only (no JSON, no markdown, no code fences). On the first line "
    "write exactly one of:\n"
    "ADVERTISER_URL: <the full URL you found>\n"
    "ADVERTISER_URL: NONE\n"
    "Only state a URL you actually found in search results - never construct or guess one; a "
    "wrong URL is worse than admitting you couldn't find one."
)

_ANSWER_LINE_RE = re.compile(r"ADVERTISER_URL:\s*(\S+)", re.IGNORECASE)


class GroundedSearchError(Exception):
    """The grounded Gemini call itself failed (network/API error)."""


def is_available() -> bool:
    """Grounding is Gemini-specific, independent of whatever get_llm_client()'s fallback chain
    happens to be using (Groq/Ollama/Mock don't support it at all)."""
    return bool(os.getenv("GEMINI_API_KEY"))


def _extract_advertiser_url(text: str) -> Optional[str]:
    """Pulls the URL out of the model's `ADVERTISER_URL: ...` line, then re-validates it against
    ADVERTISER_URL_RE so a malformed or NONE answer can't slip through as a real URL."""
    match = _ANSWER_LINE_RE.search(text)
    if not match:
        return None
    candidate = match.group(1).rstrip(").,;'\"")
    url_match = ADVERTISER_URL_RE.match(candidate)
    return url_match.group(0) if url_match else None


class GeminiGroundedClient:
    """One Google-Search-grounded Gemini call."""

    def __init__(self, genai_client, model: str):
        self._client = genai_client
        self._model = model

    def search(self, user_prompt: str) -> str:
        """Raises GroundedSearchError on any request failure."""
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=GROUNDING_SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        try:
            response = self._client.models.generate_content(model=self._model, contents=user_prompt, config=config)
            return response.text or ""
        except (errors.ClientError, errors.ServerError) as e:
            raise GroundedSearchError(str(e)) from e
        except Exception as e:  # noqa: BLE001 - any SDK/network failure here just means "skip this brand"
            raise GroundedSearchError(str(e)) from e


def find_advertiser_url(brand_name: str, domain: Optional[str] = None) -> Optional[str]:
    """Never raises. Returns None whenever grounding isn't available, the call failed, or no
    valid advertiser URL could be extracted - callers treat every None identically."""
    if not is_available():
        return None

    from google import genai
    from google.genai import types

    gemini_key = os.getenv("GEMINI_API_KEY")
    model = _parse_model_list(os.getenv("GEMINI_MODELS", DEFAULT_GEMINI_MODELS))[0]
    genai_client = genai.Client(api_key=gemini_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    client = GeminiGroundedClient(genai_client, model)

    user_prompt = f"Brand: {brand_name}" + (f"\nKnown domain: {domain}" if domain else "")
    try:
        raw = client.search(user_prompt)
    except GroundedSearchError:
        return None

    return _extract_advertiser_url(raw)
