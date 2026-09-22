"""Tests for GeminiClient's quota-exhaustion fallback, using a mocked API - no real calls."""
import os
from unittest.mock import MagicMock, patch

import pytest

from core.reasoning.llm_client import GeminiClient, QuotaExceededError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    return GeminiClient(model="primary-model", fallback_model="fallback-model")


def _quota_error():
    from google.genai import errors

    return errors.ClientError(429, {"error": {"message": "quota exceeded"}})


def test_falls_back_to_second_model_when_primary_quota_exhausted(client):
    primary_response = MagicMock()
    fallback_response = MagicMock(text="fallback output")

    with patch.object(client._client.models, "generate_content") as mock_generate:
        mock_generate.side_effect = [_quota_error(), fallback_response]
        result = client.generate("system", "user")

    assert result == "fallback output"
    assert mock_generate.call_count == 2
    assert mock_generate.call_args_list[0].kwargs["model"] == "primary-model"
    assert mock_generate.call_args_list[1].kwargs["model"] == "fallback-model"


def test_raises_if_no_fallback_configured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    client_without_fallback = GeminiClient(model="primary-model", fallback_model=None)

    with patch.object(client_without_fallback._client.models, "generate_content") as mock_generate:
        mock_generate.side_effect = _quota_error()
        with pytest.raises(QuotaExceededError):
            client_without_fallback.generate("system", "user")
