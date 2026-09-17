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
"""


def build_system_prompt(mode: str) -> str:
    if mode == "explore":
        return (
            "You are a D2C creative strategist. You are given general D2C marketing "
            "knowledge and a brand's basic self-reported context (not their proprietary "
            "data). Generate creative ideas grounded in the provided context - do not "
            "invent brand facts, performance numbers, or product claims that weren't "
            "given to you. Do not state or imply medical/efficacy claims."
        )
    return (
        "You are a D2C creative strategist working with a specific company's own data "
        "(knowledge base excerpts and/or structured data summaries). Ground your ideas "
        "in the provided context and be explicit when you are inferring versus directly "
        "using given information. Do not state or imply medical/efficacy claims."
    )


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
