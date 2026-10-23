"""Google Ads Transparency Center ad search, via Apify's hosted `silva95gustavo/google-ads-
scraper` actor (https://apify.com/silva95gustavo/google-ads-scraper) - a paid, pay-per-event
third-party scraper (~$0.90/1,000 ads), not an official Google API. Google publishes no public
API for ordinary commercial ads the way Meta does for its Ad Library, and this actor's own docs
are explicit that it only works against a *specific* advertiser-detail URL
(adstransparency.google.com/advertiser/AR<id>/...), not a brand name or domain - resolving a
brand to that URL is a separate step (core.reasoning.grounded_search), not this module's job.

This module has no idea what a "brand" is - callers hand it an already-resolved advertiser URL
and get ads back. Scraping a third party's UI this way is inherently more fragile than Meta's
official API (core.ad_library): actor behavior, pricing, and the underlying site's own terms of
service are all outside this project's control. That trade-off was made deliberately, to get
Google/YouTube ad coverage Meta's API doesn't have - see README's "Competitor ad research"
section.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

APIFY_ACTOR_ENDPOINT = "https://api.apify.com/v2/actors/silva95gustavo~google-ads-scraper/run-sync-get-dataset-items"
# A hair over Apify's own 300s sync-endpoint timeout, so Apify's own graceful timeout response
# wins the race instead of httpx cutting the connection first.
APIFY_REQUEST_TIMEOUT = 310.0

DEFAULT_RESULTS_LIMIT = int(os.getenv("GOOGLE_ADS_RESULTS_LIMIT", "15"))

# The one URL shape the Apify actor accepts as input - also the contract
# core.reasoning.grounded_search's find_advertiser_url() must satisfy.
ADVERTISER_URL_RE = re.compile(r"https://adstransparency\.google\.com/advertiser/AR[0-9A-Za-z]+(?:/\S*)?")


def is_valid_advertiser_url(url: str) -> bool:
    return bool(ADVERTISER_URL_RE.fullmatch(url))


class GoogleAdsError(Exception):
    """The Apify actor run failed - bad token, rate limit, or a non-2xx/malformed response."""


@dataclass
class GoogleAdExample:
    """One ad found on a brand's Google Ads Transparency Center page - inspiration, not a
    template to copy verbatim (same framing as core.ad_library.AdExample)."""

    advertiser_id: str
    advertiser_name: str
    format: str = "UNKNOWN"  # "TEXT" | "IMAGE" | "VIDEO" | "UNKNOWN"
    first_shown: Optional[date] = None
    last_shown: Optional[date] = None
    preview_url: str = ""
    headline_or_description: str = ""
    is_active: bool = False


@dataclass
class GoogleAdSearchResult:
    ads: list[GoogleAdExample] = field(default_factory=list)
    error: Optional[str] = None


def is_configured() -> bool:
    return bool(os.getenv("APIFY_API_TOKEN"))


def _parse_apify_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _to_google_ad_example(raw: dict) -> GoogleAdExample:
    first_shown = _parse_apify_date(raw.get("firstShown"))
    last_shown = _parse_apify_date(raw.get("lastShown"))
    # No explicit "still running" flag in this actor's output (unlike Meta's ad_delivery_stop_time)
    # - treat an ad last seen within the last week as the closest available proxy for active.
    is_active = bool(last_shown and (date.today() - last_shown).days <= 7)

    variations = raw.get("variations") or []
    first_variation = variations[0] if variations else {}
    text = first_variation.get("description") or first_variation.get("clickUrl") or "(no ad text available)"

    return GoogleAdExample(
        advertiser_id=raw.get("advertiserId", ""),
        advertiser_name=raw.get("advertiserName", "Unknown advertiser"),
        format=raw.get("format", "UNKNOWN"),
        first_shown=first_shown,
        last_shown=last_shown,
        preview_url=raw.get("previewUrl", ""),
        headline_or_description=text,
        is_active=is_active,
    )


class ApifyGoogleAdsClient:
    """One scrape of one advertiser's page via Apify. Stateless - safe to share across every
    session (same reasoning as MetaAdLibraryClient in core.ad_library)."""

    def __init__(self, api_token: str):
        self._api_token = api_token

    def scrape_advertiser(
        self, advertiser_url: str, results_limit: int = DEFAULT_RESULTS_LIMIT
    ) -> list[GoogleAdExample]:
        """Raises ValueError (no HTTP call made, no Apify credit spent) if advertiser_url isn't
        a valid Ads Transparency Center advertiser URL. Raises GoogleAdsError on any request
        failure."""
        if not is_valid_advertiser_url(advertiser_url):
            raise ValueError(f"Not a valid Google Ads Transparency Center advertiser URL: {advertiser_url!r}")

        import httpx

        payload = {
            "startUrls": [{"url": advertiser_url}],
            "resultsLimit": results_limit,
            "skipDetails": True,
        }
        try:
            response = httpx.post(
                APIFY_ACTOR_ENDPOINT,
                headers={"Authorization": f"Bearer {self._api_token}"},
                json=payload,
                timeout=APIFY_REQUEST_TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise GoogleAdsError(f"Couldn't reach Apify's Google Ads Scraper actor: {e}") from e

        if response.status_code != 200 and response.status_code != 201:
            try:
                message = response.json().get("error", {}).get("message", response.text[:300])
            except ValueError:
                message = response.text[:300]
            raise GoogleAdsError(f"Apify actor run returned {response.status_code}: {message}")

        try:
            data = response.json()
        except ValueError as e:
            raise GoogleAdsError(f"Apify actor returned a non-JSON response: {e}") from e

        if not isinstance(data, list):
            raise GoogleAdsError(f"Apify actor returned an unexpected shape: {type(data).__name__}")

        return [_to_google_ad_example(raw) for raw in data]


def get_google_ads_client() -> Optional[ApifyGoogleAdsClient]:
    """None if APIFY_API_TOKEN isn't configured - callers should skip the Google side gracefully
    rather than show a confusing error."""
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        return None
    return ApifyGoogleAdsClient(token)


def search_google_ads(advertiser_url: str, results_limit: int = DEFAULT_RESULTS_LIMIT) -> GoogleAdSearchResult:
    """The one function other modules should call for an already-resolved advertiser URL.
    Never raises."""
    client = get_google_ads_client()
    if client is None:
        return GoogleAdSearchResult(error="Google ad search isn't set up - add APIFY_API_TOKEN to use it.")
    try:
        return GoogleAdSearchResult(ads=client.scrape_advertiser(advertiser_url, results_limit=results_limit))
    except (GoogleAdsError, ValueError) as e:
        return GoogleAdSearchResult(error=str(e))
