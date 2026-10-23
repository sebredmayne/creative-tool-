"""Meta Ad Library search via Apify's hosted `curious_coder/facebook-ads-library-scraper`
actor - an alternative to core.ad_library's official Graph API path, for when
META_AD_LIBRARY_ACCESS_TOKEN isn't configured. Getting that official token requires a Meta
Developer App plus App Review, which is slow and gated; this gives an immediate alternative
using the same APIFY_API_TOKEN already set up for core/google_ads.py.

Prefer core.ad_library's official API wherever it's configured - it's free, official, and has
no scraping/ToS risk, both of which this module gives up by scraping Meta's own Ad Library
website instead. This exists only because the official token is genuinely harder to obtain, not
because scraping is the better approach - see core/meta_ads.py, which picks whichever of the two
is actually available and always prefers this module last.

Unlike core.google_ads's actor (which needs a specific advertiser URL resolved first), this
actor accepts a plain keyword search the same way Meta's own API does - the Ad Library's search
URL takes a `q=<keyword>` parameter directly, so no brand-to-URL resolution step is needed here.

Field names below are this module's best current understanding of this third-party actor's
output shape - Apify doesn't publish a literal example output row for it, only field
descriptions, so this was not verified against a real response at the time of writing. If ads
start coming back with empty text/dates, check the actual returned JSON keys before assuming
the request itself is broken (same honesty caveat core.ad_library's docstring already applies
to Meta's own official API).

Confirmed in practice: this actor's keyword search is inconsistent - some brands come back
cleanly matched, others come back with completely unrelated ads (seen live: searching "Complan"
returned an accounting masterclass ad and a US LLC formation ad, neither mentioning Complan at
all). Meta's own official Graph API doesn't have this problem, so _filter_relevant_ads below is
a defensive, scraper-specific correction, not something core.ad_library needs.
"""
import os
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import quote

from core.ad_library import AdExample, AdSearchResult, DEFAULT_COUNTRY

APIFY_ACTOR_ENDPOINT = (
    "https://api.apify.com/v2/actors/curious_coder~facebook-ads-library-scraper/run-sync-get-dataset-items"
)
APIFY_REQUEST_TIMEOUT = 310.0
DEFAULT_RESULTS_LIMIT = int(os.getenv("META_ADS_APIFY_RESULTS_LIMIT", "15"))


class MetaAdsApifyError(Exception):
    """The Apify actor run failed - bad token, rate limit, or a non-2xx/malformed response."""


def is_configured() -> bool:
    return bool(os.getenv("APIFY_API_TOKEN"))


def _build_search_url(query: str, country: str) -> str:
    return (
        "https://www.facebook.com/ads/library/?active_status=active&ad_type=all"
        f"&country={country}&q={quote(query)}&search_type=keyword_unordered&media_type=all"
    )


def _parse_meta_date(value) -> Optional[date]:
    """Handles both a Unix timestamp (int/float - the shape Meta's internal Ad Library API
    itself uses for start_date/end_date) and an ISO date string, since this actor's exact
    output type for these fields isn't confirmed (see module docstring)."""
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value).date()
        except (ValueError, OSError, OverflowError):
            return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _to_ad_example(raw: dict) -> AdExample:
    start_date = _parse_meta_date(raw.get("start_date", raw.get("startDate")))
    end_date = _parse_meta_date(raw.get("end_date", raw.get("endDate")))
    is_active = bool(raw.get("is_active", raw.get("isActive", end_date is None)))
    days_running = (date.today() - start_date).days if (start_date and is_active) else None

    snapshot = raw.get("snapshot")
    body = snapshot.get("body") if isinstance(snapshot, dict) else None
    body_text = (
        (body.get("text") if isinstance(body, dict) else None)
        or raw.get("ad_creative_body")
        or "(no ad text available)"
    )

    platforms_raw = raw.get("publisher_platform", raw.get("publisherPlatform")) or []
    platforms = list(platforms_raw) if isinstance(platforms_raw, list) else [str(platforms_raw)]

    ad_id = str(raw.get("ad_archive_id", raw.get("adArchiveID", raw.get("id", ""))))
    snapshot_url = f"https://www.facebook.com/ads/library/?id={ad_id}" if ad_id else ""

    return AdExample(
        ad_id=ad_id,
        page_name=raw.get("page_name", raw.get("pageName", "Unknown page")),
        body_text=body_text,
        snapshot_url=snapshot_url,
        platforms=platforms,
        start_date=start_date,
        days_running=days_running,
        is_active=is_active,
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_relevant(page_name: str, query: str) -> bool:
    """A page name counts as relevant if it or the search query contains the other, once both
    are lowercased and stripped to alphanumerics (so "Complan" matches "Complan India" or
    "COMPLAN®"). Deliberately simple rather than fuzzy-matched: this only needs to catch the
    actor returning ads for a completely unrelated page, not to rank close variants."""
    normalized_page, normalized_query = _normalize(page_name), _normalize(query)
    if not normalized_page or not normalized_query:
        return False
    return normalized_query in normalized_page or normalized_page in normalized_query


class ApifyMetaAdsClient:
    """One search via Apify's Facebook Ad Library actor. Stateless - safe to share across every
    session (same reasoning as MetaAdLibraryClient in core.ad_library)."""

    def __init__(self, api_token: str):
        self._api_token = api_token

    def search(
        self, query: str, country: str = DEFAULT_COUNTRY, results_limit: int = DEFAULT_RESULTS_LIMIT
    ) -> list[AdExample]:
        """Raises MetaAdsApifyError on any request failure."""
        import httpx

        payload = {
            "urls": [{"url": _build_search_url(query, country)}],
            "count": results_limit,
            "scrapePageAds.activeStatus": "active",
            "scrapePageAds.countryCode": country,
        }
        try:
            response = httpx.post(
                APIFY_ACTOR_ENDPOINT,
                headers={"Authorization": f"Bearer {self._api_token}"},
                json=payload,
                timeout=APIFY_REQUEST_TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise MetaAdsApifyError(f"Couldn't reach Apify's Facebook Ad Library actor: {e}") from e

        if response.status_code not in (200, 201):
            try:
                message = response.json().get("error", {}).get("message", response.text[:300])
            except ValueError:
                message = response.text[:300]
            raise MetaAdsApifyError(f"Apify actor run returned {response.status_code}: {message}")

        try:
            data = response.json()
        except ValueError as e:
            raise MetaAdsApifyError(f"Apify actor returned a non-JSON response: {e}") from e

        if not isinstance(data, list):
            raise MetaAdsApifyError(f"Apify actor returned an unexpected shape: {type(data).__name__}")

        ads = [_to_ad_example(raw) for raw in data[:results_limit]]
        # This actor's keyword search sometimes returns ads for a page with no relation to the
        # query at all (see module docstring) - drop those rather than show a marketer someone
        # else's ad as if it were the brand they searched for.
        ads = [ad for ad in ads if _is_relevant(ad.page_name, query)]
        ads.sort(key=lambda ad: ad.days_running or -1, reverse=True)
        return ads


def get_apify_meta_ads_client() -> Optional[ApifyMetaAdsClient]:
    """None if APIFY_API_TOKEN isn't configured - callers should skip this backend gracefully."""
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        return None
    return ApifyMetaAdsClient(token)


def search_meta_ads_via_apify(
    query: str, country: str = DEFAULT_COUNTRY, results_limit: int = DEFAULT_RESULTS_LIMIT
) -> AdSearchResult:
    """The one function other modules should call. Never raises."""
    client = get_apify_meta_ads_client()
    if client is None:
        return AdSearchResult(error="Meta ad search via Apify isn't set up - add APIFY_API_TOKEN to use it.")
    try:
        return AdSearchResult(ads=client.search(query, country=country, results_limit=results_limit))
    except MetaAdsApifyError as e:
        return AdSearchResult(error=str(e))
