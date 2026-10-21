"""Tests for research_category's orchestration: every call it makes downstream is patched, so
these exercise only the fan-out/fan-in logic and the "one brand's failure never affects another
brand or the other platform" guarantee."""
from unittest.mock import patch

import pytest

from core.ad_library import AdExample, AdSearchResult
from core.brand_discovery import BrandDiscoveryError, BrandDiscoveryResult, CompetitorBrand
from core.competitor_research import research_category
from core.google_ads import GoogleAdExample, GoogleAdSearchResult
from core.reasoning.trends_synthesis import TrendsSummary, TrendsSynthesisError


def _brand(name):
    return CompetitorBrand(brand_name=name, domain=f"{name.lower()}.com")


def _meta_ad(brand_name):
    return AdExample(ad_id="1", page_name=brand_name, body_text="Buy now", snapshot_url="https://facebook.com/ads/1")


def _google_ad(brand_name):
    return GoogleAdExample(advertiser_id="AR1", advertiser_name=brand_name, headline_or_description="Buy now")


_FAKE_SUMMARY = TrendsSummary(overall_summary="Everyone focuses on convenience.")


@pytest.fixture
def llm_client():
    return object()  # never actually called directly by research_category - only by patched fns


def _patch_pipeline(**overrides):
    """Patches every function research_category calls downstream, with sensible happy-path
    defaults overridable per test."""
    defaults = dict(
        discover_top_brands=lambda category, llm_client, count=5: BrandDiscoveryResult(
            category_normalized=category, brands=[_brand("Acme"), _brand("Bravo")]
        ),
        search_ads=lambda query: AdSearchResult(ads=[_meta_ad(query)]),
        google_ads_configured=lambda: True,
        grounding_available=lambda: True,
        find_advertiser_url=lambda brand_name, domain=None: f"https://adstransparency.google.com/advertiser/AR{brand_name}",
        search_google_ads=lambda url: GoogleAdSearchResult(ads=[_google_ad(url)]),
        synthesize_trends=lambda category, summaries, llm_client: _FAKE_SUMMARY,
    )
    defaults.update(overrides)
    patches = [patch(f"core.competitor_research.{name}", fn) for name, fn in defaults.items()]
    for p in patches:
        p.start()
    return patches


def _stop_all(patches):
    for p in patches:
        p.stop()


# --- happy path --------------------------------------------------------------------------


def test_happy_path_returns_populated_result_for_every_brand(llm_client):
    patches = _patch_pipeline()
    try:
        result = research_category("milk mix", llm_client)
    finally:
        _stop_all(patches)

    assert result.error is None
    assert len(result.brands) == 2
    assert all(b.meta_ads for b in result.brands)
    assert all(b.google_ads for b in result.brands)
    assert all(b.google_skipped_reason is None for b in result.brands)
    assert result.trends_summary is _FAKE_SUMMARY


# --- total failure: brand discovery itself fails ------------------------------------------


def test_discovery_failure_short_circuits_with_nothing_else_called(llm_client):
    def _raise_discovery(category, llm_client, count=5):
        raise BrandDiscoveryError("model returned garbage")

    search_ads_calls = []
    patches = _patch_pipeline(
        discover_top_brands=_raise_discovery,
        search_ads=lambda query: search_ads_calls.append(query) or AdSearchResult(),
    )
    try:
        result = research_category("milk mix", llm_client)
    finally:
        _stop_all(patches)

    assert search_ads_calls == []  # discovery failed before any brand was ever researched
    assert result.error is not None
    assert result.brands == []


def test_empty_brand_list_is_treated_as_an_error(llm_client):
    patches = _patch_pipeline(
        discover_top_brands=lambda category, llm_client, count=5: BrandDiscoveryResult(
            category_normalized=category, brands=[]
        )
    )
    try:
        result = research_category("an extremely obscure category", llm_client)
    finally:
        _stop_all(patches)

    assert result.error is not None


# --- per-brand Google failures must not affect Meta results or other brands ---------------


def test_one_brand_failing_google_resolution_does_not_affect_other_brands_or_meta(llm_client):
    def _find_url(brand_name, domain=None):
        return None if brand_name == "Bravo" else f"https://adstransparency.google.com/advertiser/AR{brand_name}"

    patches = _patch_pipeline(find_advertiser_url=_find_url)
    try:
        result = research_category("milk mix", llm_client)
    finally:
        _stop_all(patches)

    by_name = {b.brand_name: b for b in result.brands}
    assert by_name["Bravo"].google_ads == []
    assert by_name["Bravo"].google_skipped_reason is not None
    assert by_name["Acme"].google_ads  # unaffected by Bravo's failure
    assert by_name["Acme"].meta_ads and by_name["Bravo"].meta_ads  # Meta unaffected on both sides


def test_google_side_entirely_skipped_when_apify_not_configured(llm_client):
    patches = _patch_pipeline(google_ads_configured=lambda: False)
    try:
        result = research_category("milk mix", llm_client)
    finally:
        _stop_all(patches)

    assert all(b.google_ads == [] for b in result.brands)
    assert all("APIFY_API_TOKEN" in b.google_skipped_reason for b in result.brands)
    assert all(b.meta_ads for b in result.brands)  # Meta still works


def test_google_side_entirely_skipped_when_grounding_unavailable(llm_client):
    patches = _patch_pipeline(grounding_available=lambda: False)
    try:
        result = research_category("milk mix", llm_client)
    finally:
        _stop_all(patches)

    assert all(b.google_ads == [] for b in result.brands)
    assert all("GEMINI_API_KEY" in b.google_skipped_reason for b in result.brands)


# --- synthesis failure leaves brand findings intact ----------------------------------------


def test_synthesis_failure_leaves_brand_findings_populated(llm_client):
    def _raise_synthesis(category, summaries, llm_client):
        raise TrendsSynthesisError("couldn't parse")

    patches = _patch_pipeline(synthesize_trends=_raise_synthesis)
    try:
        result = research_category("milk mix", llm_client)
    finally:
        _stop_all(patches)

    assert result.trends_summary is None
    assert result.error is None
    assert len(result.brands) == 2
    assert all(b.meta_ads for b in result.brands)


# --- every discovered brand gets processed, even if one blows up unexpectedly -------------


def test_all_brands_processed_even_if_one_raises_unexpectedly(llm_client):
    def _search_ads(query):
        if query == "Bravo":
            raise RuntimeError("boom")
        return AdSearchResult(ads=[_meta_ad(query)])

    patches = _patch_pipeline(search_ads=_search_ads)
    try:
        result = research_category("milk mix", llm_client)
    finally:
        _stop_all(patches)

    assert len(result.brands) == 2
    by_name = {b.brand_name: b for b in result.brands}
    assert by_name["Acme"].meta_ads  # unaffected by Bravo's crash
    assert by_name["Bravo"].meta_error is not None
