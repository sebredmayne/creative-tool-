"""Tests for core.meta_ads's official-vs-Apify-fallback selection logic. Both backends are
patched - this only exercises the picking logic itself."""
from unittest.mock import patch

from core.ad_library import AdSearchResult
from core.meta_ads import is_configured, search_meta_ads


def test_is_configured_true_when_only_official_token_set(monkeypatch):
    monkeypatch.setenv("META_AD_LIBRARY_ACCESS_TOKEN", "fake-token")
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    assert is_configured() is True


def test_is_configured_true_when_only_apify_token_set(monkeypatch):
    monkeypatch.delenv("META_AD_LIBRARY_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")
    assert is_configured() is True


def test_is_configured_false_when_neither_set(monkeypatch):
    monkeypatch.delenv("META_AD_LIBRARY_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    assert is_configured() is False


def test_prefers_official_api_when_both_are_configured(monkeypatch):
    monkeypatch.setenv("META_AD_LIBRARY_ACCESS_TOKEN", "fake-token")
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-apify-token")

    with patch("core.meta_ads.search_ads", return_value=AdSearchResult()) as mock_official, patch(
        "core.meta_ads.search_meta_ads_via_apify"
    ) as mock_apify:
        search_meta_ads("skincare")

    mock_official.assert_called_once_with("skincare")
    mock_apify.assert_not_called()


def test_falls_back_to_apify_when_official_token_is_missing(monkeypatch):
    monkeypatch.delenv("META_AD_LIBRARY_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-apify-token")

    with patch("core.meta_ads.search_ads") as mock_official, patch(
        "core.meta_ads.search_meta_ads_via_apify", return_value=AdSearchResult()
    ) as mock_apify:
        search_meta_ads("skincare")

    mock_official.assert_not_called()
    mock_apify.assert_called_once_with("skincare")


def test_returns_a_plain_message_when_neither_backend_is_configured(monkeypatch):
    monkeypatch.delenv("META_AD_LIBRARY_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)

    result = search_meta_ads("skincare")

    assert result.ads == []
    assert "META_AD_LIBRARY_ACCESS_TOKEN" in result.error
    assert "APIFY_API_TOKEN" in result.error
