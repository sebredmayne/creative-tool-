"""Tests for the Apify-backed Google Ads Transparency client - mocked HTTP responses
throughout, no real network calls and no real Apify token needed."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.google_ads import (
    ApifyGoogleAdsClient,
    GoogleAdsError,
    get_google_ads_client,
    is_configured,
    is_valid_advertiser_url,
    search_google_ads,
)

_VALID_URL = "https://adstransparency.google.com/advertiser/AR01234567890123456789"


def _raw_ad(advertiser_id="AR123", days_ago=2, description="Buy our milk mix"):
    last_shown = date.today() - timedelta(days=days_ago)
    return {
        "advertiserId": advertiser_id,
        "advertiserName": "Acme Milk Co",
        "format": "TEXT",
        "firstShown": (last_shown - timedelta(days=30)).isoformat(),
        "lastShown": last_shown.isoformat(),
        "previewUrl": "https://adstransparency.google.com/advertiser/AR123/creative/CR123",
        "variations": [{"description": description, "clickUrl": "https://acmemilk.com"}],
    }


def _mock_response(status_code=200, json_data=None, text=""):
    response = MagicMock(status_code=status_code, text=text)
    response.json.return_value = json_data if json_data is not None else []
    return response


# --- is_valid_advertiser_url ----------------------------------------------------------------


def test_valid_advertiser_url_accepted():
    assert is_valid_advertiser_url(_VALID_URL) is True


def test_random_url_rejected():
    assert is_valid_advertiser_url("https://example.com/not-an-advertiser-page") is False


def test_google_search_results_url_rejected():
    assert is_valid_advertiser_url("https://adstransparency.google.com/?region=IN&domain=example.com") is False


# --- ApifyGoogleAdsClient.scrape_advertiser: parsing, errors --------------------------------


def test_scrape_advertiser_parses_ad_fields_correctly():
    response = _mock_response(json_data=[_raw_ad(days_ago=1)])
    with patch("httpx.post", return_value=response):
        client = ApifyGoogleAdsClient(api_token="fake-token")
        ads = client.scrape_advertiser(_VALID_URL)

    assert len(ads) == 1
    ad = ads[0]
    assert ad.advertiser_id == "AR123"
    assert ad.advertiser_name == "Acme Milk Co"
    assert ad.format == "TEXT"
    assert ad.headline_or_description == "Buy our milk mix"
    assert ad.is_active is True


def test_ad_last_shown_long_ago_is_not_active():
    response = _mock_response(json_data=[_raw_ad(days_ago=90)])
    with patch("httpx.post", return_value=response):
        client = ApifyGoogleAdsClient(api_token="fake-token")
        ads = client.scrape_advertiser(_VALID_URL)

    assert ads[0].is_active is False


def test_falls_back_to_click_url_when_no_description_present():
    raw = _raw_ad()
    raw["variations"] = [{"clickUrl": "https://acmemilk.com/offer"}]
    response = _mock_response(json_data=[raw])
    with patch("httpx.post", return_value=response):
        client = ApifyGoogleAdsClient(api_token="fake-token")
        ads = client.scrape_advertiser(_VALID_URL)

    assert ads[0].headline_or_description == "https://acmemilk.com/offer"


def test_scrape_advertiser_raises_value_error_for_invalid_url_without_any_http_call():
    client = ApifyGoogleAdsClient(api_token="fake-token")
    with patch("httpx.post") as mock_post:
        with pytest.raises(ValueError):
            client.scrape_advertiser("https://example.com/not-an-advertiser")
        mock_post.assert_not_called()


def test_scrape_advertiser_sends_the_required_request_shape():
    response = _mock_response(json_data=[])
    with patch("httpx.post", return_value=response) as mock_post:
        client = ApifyGoogleAdsClient(api_token="fake-token")
        client.scrape_advertiser(_VALID_URL, results_limit=10)

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer fake-token"
    assert kwargs["json"]["startUrls"] == [{"url": _VALID_URL}]
    assert kwargs["json"]["resultsLimit"] == 10
    assert kwargs["json"]["skipDetails"] is True


def test_raises_google_ads_error_on_non_200_response():
    response = _mock_response(status_code=400, json_data={"error": {"message": "Invalid token"}})
    with patch("httpx.post", return_value=response):
        client = ApifyGoogleAdsClient(api_token="bad-token")
        with pytest.raises(GoogleAdsError, match="Invalid token"):
            client.scrape_advertiser(_VALID_URL)


def test_raises_google_ads_error_when_error_body_is_not_json():
    response = _mock_response(status_code=500, text="Internal Server Error")
    response.json.side_effect = ValueError("not json")
    with patch("httpx.post", return_value=response):
        client = ApifyGoogleAdsClient(api_token="fake-token")
        with pytest.raises(GoogleAdsError):
            client.scrape_advertiser(_VALID_URL)


def test_raises_google_ads_error_on_unexpected_shape():
    response = _mock_response(json_data={"not": "a list"})
    with patch("httpx.post", return_value=response):
        client = ApifyGoogleAdsClient(api_token="fake-token")
        with pytest.raises(GoogleAdsError):
            client.scrape_advertiser(_VALID_URL)


def test_raises_google_ads_error_on_network_failure():
    import httpx

    with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
        client = ApifyGoogleAdsClient(api_token="fake-token")
        with pytest.raises(GoogleAdsError):
            client.scrape_advertiser(_VALID_URL)


# --- is_configured / get_google_ads_client / search_google_ads: the "not set up" path -------


def test_is_configured_false_without_token(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    assert is_configured() is False


def test_is_configured_true_with_token(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")
    assert is_configured() is True


def test_get_google_ads_client_is_none_without_token(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    assert get_google_ads_client() is None


def test_search_google_ads_returns_a_plain_message_when_not_configured(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = search_google_ads(_VALID_URL)
    assert result.ads == []
    assert result.error is not None
    assert "APIFY_API_TOKEN" in result.error


def test_search_google_ads_never_raises_on_invalid_url(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")
    result = search_google_ads("https://example.com/not-an-advertiser")
    assert result.ads == []
    assert result.error is not None


def test_search_google_ads_never_raises_on_api_failure(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")
    response = _mock_response(status_code=400, json_data={"error": {"message": "rate limited"}})
    with patch("httpx.post", return_value=response):
        result = search_google_ads(_VALID_URL)

    assert result.ads == []
    assert "rate limited" in result.error


def test_search_google_ads_returns_ads_on_success(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")
    response = _mock_response(json_data=[_raw_ad()])
    with patch("httpx.post", return_value=response):
        result = search_google_ads(_VALID_URL)

    assert result.error is None
    assert len(result.ads) == 1
