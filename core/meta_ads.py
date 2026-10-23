"""Picks between core.ad_library's official Meta Ad Library API (preferred - free, official, no
scraping/ToS risk) and core.meta_ads_apify's Apify-scraped fallback (used only when the official
META_AD_LIBRARY_ACCESS_TOKEN isn't configured, since getting one requires a Meta Developer App
and App Review that not every setup has been through). Both UI surfaces - ui/ad_inspiration.py
and ui/competitor_research.py - should go through this module rather than picking a backend
themselves, so "which Meta backend is active" is decided in exactly one place.
"""
from core.ad_library import AdSearchResult, is_configured as _official_configured, search_ads
from core.meta_ads_apify import is_configured as _apify_configured, search_meta_ads_via_apify


def is_configured() -> bool:
    return _official_configured() or _apify_configured()


def search_meta_ads(query: str) -> AdSearchResult:
    """Never raises. Prefers the official API; falls back to the Apify scraper only if the
    official token isn't set at all."""
    if _official_configured():
        return search_ads(query)
    if _apify_configured():
        return search_meta_ads_via_apify(query)
    return AdSearchResult(
        error="Meta ad search isn't set up - add META_AD_LIBRARY_ACCESS_TOKEN (official, preferred) "
        "or APIFY_API_TOKEN (scraped fallback)."
    )
