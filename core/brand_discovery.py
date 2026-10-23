"""One LLM call that identifies the top competing brands in a category, as the entry point for
core.competitor_research's "I want to make an ad for <category>" flow - the marketer names a
category/product, not a brand, and this figures out who's worth looking at.

Deliberately not grounded (see core.reasoning.grounded_search for the module that is): naming
the handful of brands that dominate a category is the kind of fact a model's training data
usually covers well enough, and grounding + structured JSON output can't be combined in one
Gemini call anyway (see core.reasoning.grounded_search's docstring). The real trade-off this
accepts: a brand that only recently started dominating paid ads in a category may be missed.
Acceptable for v1 - grounded_search.find_advertiser_url already proves the grounded path works,
so swapping this step to grounded search later is a bounded change, not a redesign.
"""
import json
from dataclasses import dataclass, field
from typing import Optional

from core.ad_library import DEFAULT_COUNTRY
from core.reasoning.engine import strip_code_fence
from core.reasoning.llm_client import LLMClient

TOP_BRANDS_SYSTEM_PROMPT = (
    "You help a D2C marketer research their competitive landscape. Given a product category, "
    "identify the brands with the strongest advertising presence in that category - the ones a "
    "marketer entering this space would actually be competing against for attention, biased "
    "toward brands with a real D2C/e-commerce presence in the given country.\n\n"
    "For each brand, give its commonly-used marketing name (not a parent holding company name "
    "unless that's also how it markets itself) and, only if you're confident of it, its primary "
    "website domain - use null for domain rather than guessing, since a wrong domain will "
    "mislead a later lookup step that depends on it.\n\n"
    "Respond with JSON only, no other text, matching this exact shape: "
    '{"category_normalized": "...", "brands": [{"brand_name": "...", "domain": "..."|null}]}'
)

TOP_BRANDS_SCHEMA = {
    "type": "object",
    "properties": {
        "category_normalized": {"type": "string"},
        "brands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "brand_name": {"type": "string"},
                    "domain": {"type": "string", "nullable": True},
                },
                "required": ["brand_name"],
            },
        },
    },
    "required": ["category_normalized", "brands"],
}


class BrandDiscoveryError(Exception):
    """The model's response couldn't be parsed into the expected shape."""


@dataclass
class CompetitorBrand:
    brand_name: str
    domain: Optional[str] = None


@dataclass
class BrandDiscoveryResult:
    category_normalized: str = ""
    brands: list[CompetitorBrand] = field(default_factory=list)


def discover_top_brands(
    category: str, llm_client: LLMClient, country: str = DEFAULT_COUNTRY, count: int = 5
) -> BrandDiscoveryResult:
    """Raises BrandDiscoveryError if the model's response can't be parsed - the caller should
    surface that as a research failure, not silently return zero brands."""
    user_prompt = (
        f"Product category: {category}\nCountry: {country}\n"
        f"Identify the top {count} brands by advertising presence in this category."
    )
    raw = llm_client.generate(TOP_BRANDS_SYSTEM_PROMPT, user_prompt, response_schema=TOP_BRANDS_SCHEMA)
    try:
        parsed = json.loads(strip_code_fence(raw))
        brands = [
            CompetitorBrand(brand_name=str(b["brand_name"]), domain=(str(b["domain"]) if b.get("domain") else None))
            for b in parsed["brands"]
        ]
        return BrandDiscoveryResult(category_normalized=str(parsed.get("category_normalized", category)), brands=brands)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise BrandDiscoveryError(f"Couldn't parse the model's brand discovery response: {e}") from e
