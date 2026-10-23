"""Meta Ad Library search - "what's running in this niche right now", as inspiration for a
brief. Deliberately not an LLM feature: no AI API involved anywhere in this module, just a
plain HTTP call to Meta's own public, official, free Ad Library API (no scraping, no ToS risk).

Real performance metrics (CTR, spend, conversions) aren't publicly available for ordinary
commercial ads - advertisers don't disclose them, and Meta doesn't expose them outside
political/social-issue ads. The one honest, industry-standard proxy available here is **how
long an ad has been running continuously**: advertisers cut losing ads fast, so a long-running
one is presumably working. That's a proxy, not a real metric, and every call site here should
keep presenting it as one - plain "most-liked" or "happiest-customer" framing elsewhere in this
app for an idea's grounding does the same (see core.models.CreativeIdea.grounding).

Field availability and the exact auth flow for Meta's Ad Library API have changed more than
once; what's implemented here reflects the author's best knowledge of the API's shape, not a
tested integration (that needs a real Meta Developer App + access token this project doesn't
have). If requests start failing, check developers.facebook.com/docs/marketing-api/ad-library-api
for what's changed before assuming this code is wrong.
"""
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

# Bump this if Meta deprecates the version (they retire old Graph API versions roughly every
# 2 years) - a "deprecated version" error from the API is the signal to check for a newer one.
GRAPH_API_VERSION = "v21.0"
AD_LIBRARY_ENDPOINT = f"https://graph.facebook.com/{GRAPH_API_VERSION}/ads_archive"

DEFAULT_COUNTRY = os.getenv("AD_LIBRARY_COUNTRY", "IN")
DEFAULT_RESULT_LIMIT = 20

# Fields requested for ordinary (non-political/social-issue) ads. Meta restricts a lot more
# (impressions, spend) to political/social-issue ads only - this list deliberately only asks
# for what's documented as available for ALL ads, so the request doesn't fail or come back
# empty due to asking for a field this app isn't authorized to see.
_REQUESTED_FIELDS = (
    "id,page_name,ad_creative_bodies,ad_creative_link_titles,ad_snapshot_url,"
    "ad_delivery_start_time,ad_delivery_stop_time,publisher_platforms"
)


class AdLibraryError(Exception):
    """The request to Meta's Ad Library API failed - bad token, rate limit, deprecated API
    version, or any other non-2xx response. Carries Meta's own error message when available."""


@dataclass
class AdExample:
    """One ad found in the library - inspiration, not a template to copy verbatim."""

    ad_id: str
    page_name: str
    body_text: str
    snapshot_url: str
    platforms: list[str] = field(default_factory=list)
    start_date: Optional[date] = None
    days_running: Optional[int] = None
    is_active: bool = False


@dataclass
class AdSearchResult:
    ads: list[AdExample] = field(default_factory=list)
    error: Optional[str] = None


def is_configured() -> bool:
    return bool(os.getenv("META_AD_LIBRARY_ACCESS_TOKEN"))


def _parse_meta_datetime(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _to_ad_example(raw: dict) -> AdExample:
    start_date = _parse_meta_datetime(raw.get("ad_delivery_start_time"))
    is_active = not raw.get("ad_delivery_stop_time")
    days_running = (date.today() - start_date).days if (start_date and is_active) else None

    bodies = raw.get("ad_creative_bodies") or []
    titles = raw.get("ad_creative_link_titles") or []
    body_text = bodies[0] if bodies else (titles[0] if titles else "(no ad text available)")

    return AdExample(
        ad_id=raw.get("id", ""),
        page_name=raw.get("page_name", "Unknown page"),
        body_text=body_text,
        snapshot_url=raw.get("ad_snapshot_url", ""),
        platforms=list(raw.get("publisher_platforms") or []),
        start_date=start_date,
        days_running=days_running,
        is_active=is_active,
    )


class MetaAdLibraryClient:
    """One search call against Meta's Ad Library API. Stateless - safe to share across every
    session (unlike CopilotSession, there's no per-user data here to isolate)."""

    def __init__(self, access_token: str, country: str = DEFAULT_COUNTRY):
        self._access_token = access_token
        self._country = country

    def search(self, query: str, limit: int = DEFAULT_RESULT_LIMIT) -> list[AdExample]:
        """Returns ads matching `query`, currently active, longest-running first (the "still
        running = probably working" proxy). Raises AdLibraryError on any request failure."""
        import httpx

        params = {
            "access_token": self._access_token,
            "search_terms": query,
            "ad_reached_countries": f'["{self._country}"]',
            "ad_active_status": "ACTIVE",
            "ad_type": "ALL",
            "limit": limit,
            "fields": _REQUESTED_FIELDS,
        }
        try:
            response = httpx.get(AD_LIBRARY_ENDPOINT, params=params, timeout=15.0)
        except httpx.HTTPError as e:
            raise AdLibraryError(f"Couldn't reach Meta's Ad Library API: {e}") from e

        if response.status_code != 200:
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError):
                message = response.text[:300]
            raise AdLibraryError(f"Meta Ad Library API returned {response.status_code}: {message}")

        data = response.json().get("data", [])
        ads = [_to_ad_example(raw) for raw in data]
        ads.sort(key=lambda ad: ad.days_running or -1, reverse=True)
        return ads


def get_ad_library_client() -> Optional[MetaAdLibraryClient]:
    """None if META_AD_LIBRARY_ACCESS_TOKEN isn't configured - callers should show a plain
    "not set up" message rather than a confusing error in that case."""
    token = os.getenv("META_AD_LIBRARY_ACCESS_TOKEN")
    if not token:
        return None
    return MetaAdLibraryClient(token, country=DEFAULT_COUNTRY)


def search_ads(query: str) -> AdSearchResult:
    """The one function the UI should call - never raises, always returns a result."""
    client = get_ad_library_client()
    if client is None:
        return AdSearchResult(error="Ad inspiration isn't set up - add META_AD_LIBRARY_ACCESS_TOKEN to use it.")
    try:
        return AdSearchResult(ads=client.search(query))
    except AdLibraryError as e:
        return AdSearchResult(error=str(e))
