"""Tests for classify_generation_error's exception -> (is_provider_side, message) mapping."""
import json

import httpx
import pytest
from google.genai import errors as genai_errors

from core.reasoning.llm_client import ChainExhaustedError, QuotaExceededError
from core.session import classify_generation_error, describe_generation_error


def test_quota_exceeded_error_names_the_daily_limit():
    message = describe_generation_error(QuotaExceededError("quota exceeded"))
    assert "quota" in message.lower()
    assert "google" not in message.lower()  # not phrased as Google's fault


def test_chain_exhausted_error_lists_every_attempted_model():
    exc = ChainExhaustedError(
        [
            ("gemini:gemini-3.6-flash", QuotaExceededError("quota")),
            ("groq:llama-3.3-70b-versatile", QuotaExceededError("quota")),
        ]
    )
    message = describe_generation_error(exc)
    assert "gemini:gemini-3.6-flash" in message
    assert "groq:llama-3.3-70b-versatile" in message


def test_server_error_blames_google():
    exc = genai_errors.ServerError(503, {"error": {"message": "overloaded"}})
    message = describe_generation_error(exc)
    assert "google" in message.lower()


def test_timeout_error_is_reported_as_a_timeout():
    message = describe_generation_error(httpx.ConnectTimeout("timed out"))
    assert "timed out" in message.lower()


def test_json_decode_error_is_reported_as_a_parse_failure():
    try:
        json.loads("not json")
    except json.JSONDecodeError as exc:
        message = describe_generation_error(exc)
    assert "pars" in message.lower()


def test_unrecognized_error_blames_the_app_not_the_provider():
    message = describe_generation_error(ValueError("something unexpected"))
    assert "google" not in message.lower()
    assert "not the model provider" in message.lower()
    assert "ValueError" in message


@pytest.mark.parametrize(
    "exc",
    [
        QuotaExceededError("quota exceeded"),
        ChainExhaustedError([("gemini:x", QuotaExceededError("quota"))]),
        genai_errors.ServerError(503, {"error": {"message": "overloaded"}}),
        httpx.ConnectTimeout("timed out"),
    ],
)
def test_provider_side_failures_are_classified_as_provider_side(exc):
    is_provider_side, _ = classify_generation_error(exc)
    assert is_provider_side is True


@pytest.mark.parametrize("exc", [ValueError("bug"), KeyError("missing")])
def test_app_bugs_are_not_classified_as_provider_side(exc):
    is_provider_side, _ = classify_generation_error(exc)
    assert is_provider_side is False
