"""Tests for the Apify-backed Meta Ad Library fallback client - mocked HTTP responses
throughout, no real network calls and no real Apify token needed."""
from unittest.mock import MagicMock, patch

import pytest

from core.meta_ads_apify import (
    ApifyMetaAdsClient,
    MetaAdsApifyError,
    _is_relevant,
    get_apify_meta_ads_client,
    is_configured,
    search_meta_ads_via_apify,
)


def _raw_ad(ad_archive_id="123", days_ago=5, active=True, page_name="Acme Co", text="Buy our thing"):
    import time

    start_ts = time.time() - days_ago * 86400
    raw = {
        "ad_archive_id": ad_archive_id,
        "page_name": page_name,
        "snapshot": {"body": {"text": text}},
        "start_date": start_ts,
        "is_active": active,
        "publisher_platform": ["facebook", "instagram"],
    }
    if not active:
        raw["end_date"] = time.time()
    return raw


def _mock_response(status_code=200, json_data=None, text=""):
    response = MagicMock(status_code=status_code, text=text)
    response.json.return_value = json_data if json_data is not None else []
    return response


# --- _is_relevant: filtering out the scraper's occasional unrelated-page results -----------


def test_exact_match_is_relevant():
    assert _is_relevant("Complan", "Complan") is True


def test_page_name_with_extra_suffix_is_relevant():
    assert _is_relevant("Complan India", "Complan") is True
    assert _is_relevant("COMPLAN®", "complan") is True


def test_completely_unrelated_page_is_not_relevant():
    assert _is_relevant("Acme Accounting Masterclass", "Complan") is False


def test_empty_page_name_is_not_relevant():
    assert _is_relevant("", "Complan") is False


def test_search_filters_out_ads_for_an_unrelated_page():
    response = _mock_response(
        json_data=[_raw_ad(page_name="Complan India"), _raw_ad(ad_archive_id="999", page_name="Unrelated Accounting Co")]
    )
    with patch("httpx.post", return_value=response):
        client = ApifyMetaAdsClient(api_token="fake-token")
        ads = client.search("Complan")

    assert len(ads) == 1
    assert ads[0].page_name == "Complan India"


# --- ApifyMetaAdsClient.search: parsing, request shape, errors -----------------------------


def test_search_parses_ad_fields_correctly():
    response = _mock_response(json_data=[_raw_ad(days_ago=5)])
    with patch("httpx.post", return_value=response):
        client = ApifyMetaAdsClient(api_token="fake-token")
        ads = client.search("Acme Co")

    assert len(ads) == 1
    ad = ads[0]
    assert ad.ad_id == "123"
    assert ad.page_name == "Acme Co"
    assert ad.body_text == "Buy our thing"
    assert ad.platforms == ["facebook", "instagram"]
    assert ad.is_active is True
    assert ad.days_running == 5


def test_inactive_ad_has_no_days_running():
    response = _mock_response(json_data=[_raw_ad(active=False, days_ago=20)])
    with patch("httpx.post", return_value=response):
        client = ApifyMetaAdsClient(api_token="fake-token")
        ads = client.search("Acme Co")

    assert ads[0].is_active is False
    assert ads[0].days_running is None


def test_falls_back_to_ad_creative_body_when_no_snapshot_present():
    raw = _raw_ad()
    del raw["snapshot"]
    raw["ad_creative_body"] = "Fallback text"
    response = _mock_response(json_data=[raw])
    with patch("httpx.post", return_value=response):
        client = ApifyMetaAdsClient(api_token="fake-token")
        ads = client.search("Acme Co")

    assert ads[0].body_text == "Fallback text"

def test_sends_the_expected_search_url_and_request_shape():
    response = _mock_response(json_data=[])
    with patch("httpx.post", return_value=response) as mock_post:
        client = ApifyMetaAdsClient(api_token="fake-token")
        client.search("cold brew coffee", country="IN", results_limit=5)

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer fake-token"
    sent_url = kwargs["json"]["urls"][0]["url"]
    assert "q=cold%20brew%20coffee" in sent_url
    assert "country=IN" in sent_url
    assert kwargs["json"]["count"] == 5


def test_raises_meta_ads_apify_error_on_non_200_response():
    response = _mock_response(status_code=400, json_data={"error": {"message": "Invalid token"}})
    with patch("httpx.post", return_value=response):
        client = ApifyMetaAdsClient(api_token="bad-token")
        with pytest.raises(MetaAdsApifyError, match="Invalid token"):
            client.search("skincare")


def test_raises_meta_ads_apify_error_on_unexpected_shape():
    response = _mock_response(json_data={"not": "a list"})
    with patch("httpx.post", return_value=response):
        client = ApifyMetaAdsClient(api_token="fake-token")
        with pytest.raises(MetaAdsApifyError):
            client.search("skincare")


def test_raises_meta_ads_apify_error_on_network_failure():
    import httpx

    with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
        client = ApifyMetaAdsClient(api_token="fake-token")
        with pytest.raises(MetaAdsApifyError):
            client.search("skincare")


# --- is_configured / get_apify_meta_ads_client / search_meta_ads_via_apify ------------------


def test_is_configured_false_without_token(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    assert is_configured() is False


def test_is_configured_true_with_token(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")
    assert is_configured() is True


def test_get_client_is_none_without_token(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    assert get_apify_meta_ads_client() is None


def test_search_via_apify_returns_a_plain_message_when_not_configured(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = search_meta_ads_via_apify("skincare")
    assert result.ads == []
    assert result.error is not None
    assert "APIFY_API_TOKEN" in result.error


def test_search_via_apify_never_raises_on_api_failure(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")
    response = _mock_response(status_code=400, json_data={"error": {"message": "rate limited"}})
    with patch("httpx.post", return_value=response):
        result = search_meta_ads_via_apify("skincare")

    assert result.ads == []
    assert "rate limited" in result.error


def test_search_via_apify_returns_ads_on_success(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "fake-token")
    response = _mock_response(json_data=[_raw_ad()])
    with patch("httpx.post", return_value=response):
        result = search_meta_ads_via_apify("Acme Co")

    assert result.error is None
    assert len(result.ads) == 1
