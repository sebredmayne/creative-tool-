"""Orchestrates one creative-generation request: route -> retrieve -> prompt -> generate -> parse.

Explore and Connect both go through this same function. The only
differences are which vector collection is queried and what context
(user-provided fields vs. uploaded company data) is available - callers are
responsible for keeping those separate (see app.py).
"""
import json

from core.models import CreativeIdea, QueryContext
from core.reasoning.llm_client import LLMClient
from core.reasoning.prompts import build_system_prompt, build_user_prompt
from core.reasoning.router import decide_route
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import VectorStore


def generate_ideas(
    context: QueryContext,
    collection_name: str,
    vector_store: VectorStore,
    structured_store: StructuredStore,
    llm_client: LLMClient,
    top_k: int = 5,
) -> list[CreativeIdea]:
    route = decide_route(context.query, structured_data_available=structured_store.has_data())

    semantic_chunks = (
        vector_store.query(collection_name, context.query, top_k=top_k) if route.use_semantic else []
    )
    structured_summary = structured_store.summarize_all() if route.use_structured else ""

    system_prompt = build_system_prompt(context.mode)
    user_prompt = build_user_prompt(context, semantic_chunks, structured_summary)

    raw_output = llm_client.generate(system_prompt, user_prompt)
    return _parse_ideas(raw_output)


def _parse_ideas(raw_output: str) -> list[CreativeIdea]:
    try:
        items = json.loads(raw_output)
        return [
            CreativeIdea(
                concept=item["concept"],
                rationale=item["rationale"],
                recommended_format=item["recommended_format"],
                source_context=item["source_context"],
                script=item.get("script"),
            )
            for item in items
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [
            CreativeIdea(
                concept="Unparsed model output",
                rationale="The model's response wasn't valid JSON in the expected shape. Showing the raw output below.",
                recommended_format="n/a",
                source_context="n/a",
                script=raw_output,
            )
        ]
