"""Prompt templates for the reasoning engine.

Kept as plain functions/strings (not a templating library) so they stay easy
to read and edit directly.
"""
from core.models import QueryContext, RetrievedChunk

OUTPUT_INSTRUCTIONS = """
Respond with a JSON array only, no other text. Each element must be an object
with these exact keys:
- "concept": short name for the creative idea
- "rationale": why this idea fits the brand/customer/objective and the context given
- "recommended_format": e.g. "Instagram Reel", "UGC video", "carousel", "static ad"
- "source_context": brief note on which knowledge or data this idea drew from
- "script": an optional short script or brief (string), or null if not applicable
- "grounding": "brand_data" if this idea is substantively grounded in the user's own
  loaded/uploaded data (chunks tagged with a source that isn't one of the general
  knowledge files), or "inference" if it mainly draws on general category knowledge
- "needs_review": true if the copy touches health, medical, efficacy, or other
  regulated-claim language that should get compliance/regulatory sign-off before use,
  false otherwise
- "review_reason": if needs_review is true, one sentence saying why; otherwise null

For video formats (Reels, UGC videos, etc.), default to writing roughly 30
seconds of content with clearly timestamped beats (hook, body, CTA) - unless
the user's request specifies a different duration, in which case follow that.
"""


_DYNAMIC_VS_FIXED_NOTE = (
    "The knowledge you're given mixes two different kinds of content. Brand voice, "
    "tone, positioning, personas, and compliance/claim rules are fixed and authoritative "
    "- follow them exactly, never reinterpret or soften a compliance rule. Hook examples "
    "and performance benchmark numbers (CAC, ROAS, etc.) are illustrative, not exhaustive "
    "or guaranteed - use them as a pattern to generate new, fresh ideas from, not as a "
    "fixed list to repeat verbatim or as facts to cite as certain. When writing a hook or "
    "a script's opening line, match the specificity and sharpness of the example hooks, "
    "not their exact wording: name a real detail, number, or ingredient, or take a clear "
    "stance, and know whether you're writing a tease (raises one doubt, resolved later) or "
    "a thesis (states the whole argument up front) per the hook-writing framework in the "
    "knowledge. Avoid generic openers - a bare demographic callout ('if you're 20-35...') "
    "or a stock trope ('what I thought vs. reality') - unless you give it a genuinely "
    "specific, non-generic detail that couldn't apply to any other brand in the category."
)


def build_system_prompt(mode: str) -> str:
    if mode == "explore":
        base = (
            "You are a D2C creative strategist. You are given general D2C marketing "
            "knowledge and a brand's basic self-reported context (not their proprietary "
            "data). Generate creative ideas grounded in the provided context - do not "
            "invent brand facts, performance numbers, or product claims that weren't "
            "given to you. Do not state or imply medical/efficacy claims."
        )
    else:
        base = (
            "You are a D2C creative strategist working with a specific company's own data "
            "(knowledge base excerpts and/or structured data summaries). Ground your ideas "
            "in the provided context and be explicit when you are inferring versus directly "
            "using given information. Do not state or imply medical/efficacy claims."
        )
    return f"{base} {_DYNAMIC_VS_FIXED_NOTE}"


def build_user_prompt(context: QueryContext, semantic_chunks: list[RetrievedChunk], structured_summary: str) -> str:
    parts = [
        "Brand context:\n"
        f"- Brand: {context.brand or 'not provided'}\n"
        f"- Product: {context.product or 'not provided'}\n"
        f"- Target customer: {context.customer or 'not provided'}\n"
        f"- Category: {context.category or 'not provided'}\n"
        f"- Objective: {context.objective or 'not provided'}"
    ]

    if semantic_chunks:
        knowledge_text = "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in semantic_chunks)
        parts.append(f"Relevant knowledge:\n{knowledge_text}")

    if structured_summary:
        parts.append(f"Relevant data summary:\n{structured_summary}")

    parts.append(f"User request: {context.query}")
    parts.append(OUTPUT_INSTRUCTIONS)

    return "\n\n".join(parts)
