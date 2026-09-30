"""One LLM call that extracts a structured brand profile from an uploaded brand guide's text.

The four extracted fields (voice_and_tone, positioning, personas, compliance_rules) are
deliberately separated - compliance ends up somewhere it can be loaded deterministically into
every request for that brand (core.knowledge.loader.load_compliance_rules), not mixed in with
illustrative brand voice that's only retrieved when it happens to rank well. The caller (the
"Add your brand" review screen) shows all four for the user to edit before anything is saved,
and requires explicit confirmation of compliance_rules specifically, since - once saved -
that's the one field injected as non-negotiable for every idea this brand generates.
"""
import json

from core.reasoning.engine import strip_code_fence
from core.reasoning.llm_client import LLMClient

EXTRACTION_SYSTEM_PROMPT = (
    "You are analyzing a brand guide document a marketer uploaded, to set up a new brand in a "
    "D2C creative-generation tool. Extract exactly four sections as plain text, each written "
    "so it can stand alone as reference material for a copywriter who has not read the "
    "original document:\n"
    "- voice_and_tone: brand archetype/personality, tone of voice, what to do and avoid "
    "stylistically.\n"
    "- positioning: product line, pricing, market differentiation, what makes this brand "
    "different from competitors.\n"
    "- personas: who the target customer(s) are.\n"
    "- compliance_rules: every claim restriction, required disclaimer, banned phrase, or "
    "regulatory/legal constraint mentioned anywhere in the document, even a single offhand "
    "sentence - err on the side of including something borderline rather than dropping it. If "
    "the document genuinely states no explicit compliance rules, say that plainly instead of "
    "leaving this blank or inventing rules that aren't actually in the document.\n\n"
    "Respond with JSON only, no other text, matching this exact shape: "
    '{"voice_and_tone": "...", "positioning": "...", "personas": "...", "compliance_rules": "..."}'
)

BRAND_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "voice_and_tone": {"type": "string"},
        "positioning": {"type": "string"},
        "personas": {"type": "string"},
        "compliance_rules": {"type": "string"},
    },
    "required": ["voice_and_tone", "positioning", "personas", "compliance_rules"],
}

_FIELDS = ("voice_and_tone", "positioning", "personas", "compliance_rules")


class BrandExtractionError(Exception):
    """The model's response couldn't be parsed into the expected shape."""


def extract_brand_profile_fields(guide_text: str, llm_client: LLMClient) -> dict:
    """Returns {"voice_and_tone": str, "positioning": str, "personas": str,
    "compliance_rules": str}. Raises BrandExtractionError if the response can't be parsed -
    the caller should surface that as a plain error, not silently save an empty profile."""
    user_prompt = f"Brand guide document:\n\n{guide_text}"
    raw = llm_client.generate(EXTRACTION_SYSTEM_PROMPT, user_prompt, response_schema=BRAND_EXTRACTION_SCHEMA)
    try:
        parsed = json.loads(strip_code_fence(raw))
        return {field_name: str(parsed[field_name]) for field_name in _FIELDS}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise BrandExtractionError(f"Couldn't parse the model's extraction response: {e}") from e
