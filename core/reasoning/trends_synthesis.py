"""One LLM call that turns a handful of brands' scraped ad samples into a readable "what's
trending in this category" summary for core.competitor_research's output panel.

Takes plain dicts, not core.competitor_research's own dataclasses - that keeps this module
decoupled from the orchestrator (which itself imports this module; dataclasses would create an
import cycle) and keeps its own tests simple to set up.

This summary is shown directly to a marketer and may end up shaping real customer-facing ad
copy. Competitor ad text in most categories routinely contains health/nutrition/efficacy
claims; the system prompt below is written to only ever attribute such claims to the
competitor who made them, never to restate or endorse them as fact - this module must not
become a second, unreviewed source of medical/efficacy claims in this app.
"""
import json
from dataclasses import dataclass, field

from core.reasoning.engine import strip_code_fence
from core.reasoning.llm_client import LLMClient

TRENDS_SYNTHESIS_SYSTEM_PROMPT = (
    "You help a D2C marketer understand what competitors are currently doing in a product "
    "category, based on samples of their actual running ads. You will be given, per brand, a "
    "handful of ad text samples pulled from Meta's Ad Library and/or Google's Ads Transparency "
    "Center.\n\n"
    "Describe only patterns actually present in the given samples - never invent ad counts, "
    "claims, or brands beyond what's given. Some ad text may contain health, nutrition, or "
    "efficacy claims (e.g. 'clinically proven', 'boosts immunity') - always attribute any such "
    "claim explicitly to the brand that made it (e.g. \"Brand X claims their product is "
    "clinically proven\"), and never restate it as established fact or as guidance yourself. "
    "You are summarizing what competitors say about themselves, not verifying or endorsing it.\n\n"
    "Respond with JSON only, no other text, matching this exact shape: "
    '{"overall_summary": "...", '
    '"per_brand_notes": [{"brand_name": "...", "note": "..."}], '
    '"patterns": ["..."], '
    '"gaps_or_opportunities": ["..."]}'
)

TRENDS_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary": {"type": "string"},
        "per_brand_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"brand_name": {"type": "string"}, "note": {"type": "string"}},
                "required": ["brand_name", "note"],
            },
        },
        "patterns": {"type": "array", "items": {"type": "string"}},
        "gaps_or_opportunities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overall_summary", "per_brand_notes", "patterns", "gaps_or_opportunities"],
}


class TrendsSynthesisError(Exception):
    """The model's response couldn't be parsed into the expected shape."""


@dataclass
class BrandNote:
    brand_name: str
    note: str


@dataclass
class TrendsSummary:
    overall_summary: str
    per_brand_notes: list[BrandNote] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    gaps_or_opportunities: list[str] = field(default_factory=list)


def synthesize_trends(category: str, brand_summaries: list[dict], llm_client: LLMClient) -> TrendsSummary:
    """`brand_summaries`: a list of plain dicts, one per brand -
    {"brand_name": str, "meta_ad_count": int, "meta_samples": list[str],
     "google_ad_count": int, "google_samples": list[str]}.
    Raises TrendsSynthesisError if the model's response can't be parsed."""
    user_prompt = f"Category: {category}\n\nBrand ad samples:\n{json.dumps(brand_summaries, indent=2)}"
    raw = llm_client.generate(TRENDS_SYNTHESIS_SYSTEM_PROMPT, user_prompt, response_schema=TRENDS_SYNTHESIS_SCHEMA)
    try:
        parsed = json.loads(strip_code_fence(raw))
        notes = [BrandNote(brand_name=str(n["brand_name"]), note=str(n["note"])) for n in parsed["per_brand_notes"]]
        return TrendsSummary(
            overall_summary=str(parsed["overall_summary"]),
            per_brand_notes=notes,
            patterns=[str(p) for p in parsed.get("patterns", [])],
            gaps_or_opportunities=[str(g) for g in parsed.get("gaps_or_opportunities", [])],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise TrendsSynthesisError(f"Couldn't parse the model's trends synthesis response: {e}") from e
