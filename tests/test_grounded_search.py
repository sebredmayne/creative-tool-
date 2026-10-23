"""Tests for the Google-Search-grounded brand -> advertiser URL resolution. Mocked genai calls
throughout, no real API calls and no real Gemini key needed."""
from unittest.mock import MagicMock, patch

import pytest

from core.reasoning.grounded_search import (
    GeminiGroundedClient,
    GroundedSearchError,
    _extract_advertiser_url,
    find_advertiser_url,
    is_available,
)

_VALID_URL = "https://adstransparency.google.com/advertiser/AR01234567890123456789"


@pytest.fixture
def genai_client():
    from google import genai

    return genai.Client(api_key="fake-key-for-tests")


# --- is_available ----------------------------------------------------------------------------


def test_is_available_false_without_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert is_available() is False


def test_is_available_true_with_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert is_available() is True


# --- _extract_advertiser_url: pulling a URL out of the model's plain-text answer -------------


def test_extracts_url_from_plain_answer_line():
    text = f"ADVERTISER_URL: {_VALID_URL}"
    assert _extract_advertiser_url(text) == _VALID_URL


def test_extracts_url_with_trailing_punctuation_stripped():
    text = f"Sure, here it is.\nADVERTISER_URL: {_VALID_URL}."
    assert _extract_advertiser_url(text) == _VALID_URL


def test_returns_none_for_explicit_none_answer():
    assert _extract_advertiser_url("ADVERTISER_URL: NONE") is None


def test_returns_none_when_answer_line_is_missing_entirely():
    assert _extract_advertiser_url("I couldn't find anything relevant.") is None


def test_returns_none_when_stated_url_does_not_match_the_expected_pattern():
    text = "ADVERTISER_URL: https://example.com/not-an-advertiser-page"
    assert _extract_advertiser_url(text) is None


# --- GeminiGroundedClient.search: request shape, error mapping ------------------------------


def test_search_returns_response_text_on_success(genai_client):
    client = GeminiGroundedClient(genai_client, model="some-model")
    with patch.object(genai_client.models, "generate_content", return_value=MagicMock(text="ok")):
        assert client.search("Acme Milk Co") == "ok"


def test_search_raises_grounded_search_error_on_client_error(genai_client):
    from google.genai import errors

    client = GeminiGroundedClient(genai_client, model="some-model")
    error = errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    with patch.object(genai_client.models, "generate_content", side_effect=error):
        with pytest.raises(GroundedSearchError):
            client.search("Acme Milk Co")


# --- find_advertiser_url: the never-raises orchestration function ---------------------------


def test_find_advertiser_url_returns_none_and_never_constructs_a_client_when_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("google.genai.Client") as mock_client_cls:
        assert find_advertiser_url("Acme Milk Co") is None
        mock_client_cls.assert_not_called()


def test_find_advertiser_url_extracts_a_valid_url(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_instance = MagicMock()
    mock_instance.models.generate_content.return_value = MagicMock(text=f"ADVERTISER_URL: {_VALID_URL}")
    with patch("google.genai.Client", return_value=mock_instance):
        assert find_advertiser_url("Acme Milk Co") == _VALID_URL


def test_find_advertiser_url_returns_none_when_model_found_nothing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_instance = MagicMock()
    mock_instance.models.generate_content.return_value = MagicMock(text="ADVERTISER_URL: NONE")
    with patch("google.genai.Client", return_value=mock_instance):
        assert find_advertiser_url("Acme Milk Co") is None


def test_find_advertiser_url_returns_none_on_request_failure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from google.genai import errors

    mock_instance = MagicMock()
    mock_instance.models.generate_content.side_effect = errors.ServerError(503, {"error": {"message": "overloaded"}})
    with patch("google.genai.Client", return_value=mock_instance):
        assert find_advertiser_url("Acme Milk Co") is None
