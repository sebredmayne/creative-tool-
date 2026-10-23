"""Tests for the Meta Ad Library client - mocked HTTP responses throughout, no real network
calls and no real access token needed."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.ad_library import (
    AdLibraryError,
    MetaAdLibraryClient,
    get_ad_library_client,
    is_configured,
    search_ads,
)


def _raw_ad(ad_id="1", days_ago=10, stopped=False, page_name="Acme Co", body="Buy our thing"):
    start = date.today() - timedelta(days=days_ago)
    raw = {
        "id": ad_id,
        "page_name": page_name,
        "ad_creative_bodies": [body],
        "ad_creative_link_titles": [],
        "ad_snapshot_url": f"https://www.facebook.com/ads/library/?id={ad_id}",
        "ad_delivery_start_time": start.isoformat(),
        "publisher_platforms": ["FACEBOOK", "INSTAGRAM"],
    }
    if stopped:
        raw["ad_delivery_stop_time"] = date.today().isoformat()
    return raw


def _mock_response(status_code=200, json_data=None, text=""):
    response = MagicMock(status_code=status_code, text=text)
    response.json.return_value = json_data or {}
    return response


# --- MetaAdLibraryClient.search: parsing, sorting, error mapping --------------------------


def test_search_parses_ad_fields_correctly():
    response = _mock_response(json_data={"data": [_raw_ad(days_ago=5)]})
    with patch("httpx.get", return_value=response):
        client = MetaAdLibraryClient(access_token="fake-token")
        ads = client.search("skincare")

    assert len(ads) == 1
    ad = ads[0]
    assert ad.ad_id == "1"
    assert ad.page_name == "Acme Co"
    assert ad.body_text == "Buy our thing"
    assert ad.platforms == ["FACEBOOK", "INSTAGRAM"]
    assert ad.is_active is True
    assert ad.days_running == 5


def test_search_sorts_by_days_running_descending():
    response = _mock_response(
        json_data={"data": [_raw_ad(ad_id="short", days_ago=3), _raw_ad(ad_id="long", days_ago=90)]}
    )
    with patch("httpx.get", return_value=response):
        client = MetaAdLibraryClient(access_token="fake-token")
        ads = client.search("skincare")

    assert [ad.ad_id for ad in ads] == ["long", "short"]


def test_stopped_ad_has_no_days_running_and_is_not_active():
    response = _mock_response(json_data={"data": [_raw_ad(stopped=True, days_ago=20)]})
    with patch("httpx.get", return_value=response):
        client = MetaAdLibraryClient(access_token="fake-token")
        ads = client.search("skincare")

    assert ads[0].is_active is False
    assert ads[0].days_running is None  # the "still running" proxy doesn't apply once it's stopped


def test_falls_back_to_link_title_when_no_ad_body_present():
    raw = _raw_ad()
    raw["ad_creative_bodies"] = []
    raw["ad_creative_link_titles"] = ["Only a headline, no body"]
    response = _mock_response(json_data={"data": [raw]})
    with patch("httpx.get", return_value=response):
        client = MetaAdLibraryClient(access_token="fake-token")
        ads = client.search("skincare")

    assert ads[0].body_text == "Only a headline, no body"


def test_search_sends_the_required_api_parameters():
    response = _mock_response(json_data={"data": []})
    with patch("httpx.get", return_value=response) as mock_get:
        client = MetaAdLibraryClient(access_token="fake-token", country="IN")
        client.search("cold brew coffee", limit=5)

    _, kwargs = mock_get.call_args
    params = kwargs["params"]
    assert params["search_terms"] == "cold brew coffee"
    assert params["ad_reached_countries"] == '["IN"]'
    assert params["ad_active_status"] == "ACTIVE"
    assert params["limit"] == 5
    assert params["access_token"] == "fake-token"


def test_raises_ad_library_error_on_non_200_response():
    response = _mock_response(status_code=400, json_data={"error": {"message": "Invalid OAuth token"}})
    with patch("httpx.get", return_value=response):
        client = MetaAdLibraryClient(access_token="bad-token")
        with pytest.raises(AdLibraryError, match="Invalid OAuth token"):
            client.search("skincare")


def test_raises_ad_library_error_when_error_body_is_not_json():
    response = _mock_response(status_code=500, text="Internal Server Error")
    response.json.side_effect = ValueError("not json")
    with patch("httpx.get", return_value=response):
        client = MetaAdLibraryClient(access_token="fake-token")
        with pytest.raises(AdLibraryError):
            client.search("skincare")


def test_raises_ad_library_error_on_network_failure():
    import httpx

    with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
        client = MetaAdLibraryClient(access_token="fake-token")
        with pytest.raises(AdLibraryError):
            client.search("skincare")


# --- is_configured / get_ad_library_client / search_ads: the "not set up" path ------------


def test_is_configured_false_without_token(monkeypatch):
    monkeypatch.delenv("META_AD_LIBRARY_ACCESS_TOKEN", raising=False)
    assert is_configured() is False


def test_is_configured_true_with_token(monkeypatch):
    monkeypatch.setenv("META_AD_LIBRARY_ACCESS_TOKEN", "fake-token")
    assert is_configured() is True


def test_get_ad_library_client_is_none_without_token(monkeypatch):
    monkeypatch.delenv("META_AD_LIBRARY_ACCESS_TOKEN", raising=False)
    assert get_ad_library_client() is None


def test_search_ads_returns_a_plain_message_when_not_configured(monkeypatch):
    monkeypatch.delenv("META_AD_LIBRARY_ACCESS_TOKEN", raising=False)
    result = search_ads("skincare")
    assert result.ads == []
    assert result.error is not None
    assert "META_AD_LIBRARY_ACCESS_TOKEN" in result.error


def test_search_ads_never_raises_on_api_failure(monkeypatch):
    monkeypatch.setenv("META_AD_LIBRARY_ACCESS_TOKEN", "fake-token")
    response = _mock_response(status_code=400, json_data={"error": {"message": "rate limited"}})
    with patch("httpx.get", return_value=response):
        result = search_ads("skincare")

    assert result.ads == []
    assert "rate limited" in result.error


def test_search_ads_returns_ads_on_success(monkeypatch):
    monkeypatch.setenv("META_AD_LIBRARY_ACCESS_TOKEN", "fake-token")
    response = _mock_response(json_data={"data": [_raw_ad()]})
    with patch("httpx.get", return_value=response):
        result = search_ads("skincare")

    assert result.error is None
    assert len(result.ads) == 1
