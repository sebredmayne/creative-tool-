"""Orchestrates the full "I want to make an ad for <category>" research flow: find the top
competing brands, pull each one's ads from Meta and Google, and synthesize a trends summary.

This is read-only research for a marketer to read - it deliberately never writes into
core.session.CopilotSession or feeds into core.reasoning.engine's idea generation. That keeps
this feature's failure modes contained: a bad brand discovery or a flaky Apify run degrades this
panel, never the main chat/generation flow.

The one rule every per-brand step follows: Meta and Google are independent per brand, and a
failure on either side for one brand must never affect another brand or the other platform. See
_research_one_brand - it has no raise statement of its own, by design.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from core.ad_library import AdExample
from core.brand_discovery import BrandDiscoveryError, CompetitorBrand, discover_top_brands
from core.google_ads import GoogleAdExample, is_configured as google_ads_configured, search_google_ads
from core.meta_ads import search_meta_ads
from core.reasoning.grounded_search import find_advertiser_url, is_available as grounding_available
from core.reasoning.llm_client import LLMClient
from core.reasoning.trends_synthesis import TrendsSummary, TrendsSynthesisError, synthesize_trends

logger = logging.getLogger("d2c_growth_copilot")

_SAMPLE_LIMIT = 5


@dataclass
class BrandAdFindings:
    brand_name: str
    domain: Optional[str] = None
    meta_ads: list[AdExample] = field(default_factory=list)
    meta_error: Optional[str] = None
    google_ads: list[GoogleAdExample] = field(default_factory=list)
    # None only when the Google side was attempted and actually returned ads - set to a
    # human-readable reason any other time, so a marketer never sees an unexplained empty state.
    google_skipped_reason: Optional[str] = None


@dataclass
class CategoryResearchResult:
    category_input: str
    category_normalized: str = ""
    brands: list[BrandAdFindings] = field(default_factory=list)
    trends_summary: Optional[TrendsSummary] = None
    # Set only on total failure (e.g. brand discovery itself failed) - individual brands'
    # Meta/Google skips are carried per-brand instead, never here.
    error: Optional[str] = None


def _research_google_side(findings: BrandAdFindings) -> None:
    """Mutates `findings` in place. Every branch here ends in either populated google_ads or a
    set google_skipped_reason - never a raise."""
    if not google_ads_configured():
        findings.google_skipped_reason = "Google ad search isn't set up - add APIFY_API_TOKEN."
        return
    if not grounding_available():
        findings.google_skipped_reason = (
            "Finding this brand's Google Ads Transparency page needs GEMINI_API_KEY (Google Search "
            "grounding is Gemini-specific)."
        )
        return

    advertiser_url = find_advertiser_url(findings.brand_name, domain=findings.domain)
    if advertiser_url is None:
        findings.google_skipped_reason = f"Couldn't find {findings.brand_name}'s Google Ads Transparency page."
        return

    result = search_google_ads(advertiser_url)
    if result.error:
        findings.google_skipped_reason = result.error
        return
    findings.google_ads = result.ads


def _research_one_brand(brand: CompetitorBrand) -> BrandAdFindings:
    """Never raises. Every sub-call it makes is already never-raising, but this still wraps
    everything in a broad except: ThreadPoolExecutor.map() re-raises a worker's exception when
    the caller iterates results, which would otherwise take down the *entire* category search
    (every brand, not just this one) over a single unexpected fault here."""
    findings = BrandAdFindings(brand_name=brand.brand_name, domain=brand.domain)
    try:
        meta_result = search_meta_ads(brand.brand_name)
        findings.meta_ads = meta_result.ads
        findings.meta_error = meta_result.error

        _research_google_side(findings)
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        logger.warning("Unexpected failure researching brand %r: %s", brand.brand_name, e)
        findings.meta_error = findings.meta_error or f"Unexpected error: {e}"
    return findings


def _summarize_for_synthesis(findings: BrandAdFindings, sample_limit: int = _SAMPLE_LIMIT) -> dict:
    return {
        "brand_name": findings.brand_name,
        "meta_ad_count": len(findings.meta_ads),
        "meta_samples": [ad.body_text for ad in findings.meta_ads[:sample_limit]],
        "google_ad_count": len(findings.google_ads),
        "google_samples": [ad.headline_or_description for ad in findings.google_ads[:sample_limit]],
    }


def research_category(category: str, llm_client: LLMClient, brand_count: int = 5) -> CategoryResearchResult:
    """Never raises. Returns a result with `error` set only if brand discovery itself failed or
    found nothing - every other failure is scoped to one brand/platform and carried in
    BrandAdFindings instead."""
    try:
        discovery = discover_top_brands(category, llm_client, count=brand_count)
    except BrandDiscoveryError as e:
        return CategoryResearchResult(category_input=category, error=f"Couldn't identify competing brands: {e}")

    if not discovery.brands:
        return CategoryResearchResult(
            category_input=category,
            category_normalized=discovery.category_normalized,
            error="Couldn't identify any brands for this category.",
        )

    # Independent per-brand I/O (one Meta call + one grounded-search call + one Apify run each) -
    # parallelized so wall-clock time tracks the slowest single brand, not the sum of all of them.
    with ThreadPoolExecutor(max_workers=len(discovery.brands)) as executor:
        brand_findings = list(executor.map(_research_one_brand, discovery.brands))

    trends_summary = None
    try:
        brand_summaries = [_summarize_for_synthesis(findings) for findings in brand_findings]
        trends_summary = synthesize_trends(discovery.category_normalized or category, brand_summaries, llm_client)
    except TrendsSynthesisError as e:
        # Per-brand findings are still useful on their own - a failed synthesis shouldn't hide them.
        logger.warning("Trends synthesis failed for category %r: %s", category, e)

    return CategoryResearchResult(
        category_input=category,
        category_normalized=discovery.category_normalized,
        brands=brand_findings,
        trends_summary=trends_summary,
    )
