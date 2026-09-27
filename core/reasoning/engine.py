"""Orchestrates one creative-generation request: route -> retrieve -> prompt -> generate -> parse.

Always draws on the curated Explore knowledge base, and additionally on
whatever's been loaded into Connect (if anything). The two are merged
rather than mutually exclusive - a question about one category shouldn't
lose its curated knowledge just because unrelated data from a different
category happens to be loaded.
"""
import json

from core.models import CreativeIdea, QueryContext
from core.reasoning.llm_client import LLMClient
from core.reasoning.prompts import build_system_prompt, build_user_prompt
from core.reasoning.router import decide_route
from core.retrieval.structured_store import StructuredStore
from core.models import RetrievedChunk
from core.retrieval.vector_store import CONNECT_COLLECTION, EXPLORE_COLLECTION, VectorStore

# Minimum number of retrieval slots reserved for Connect data (when loaded and relevant).
# Without this, a large Explore knowledge base can crowd Connect out entirely - not because
# it's less relevant, but because well-written prose chunks tend to score marginally better
# than flatter tabular row-chunks even when the tabular data is the better answer.
MIN_CONNECT_SLOTS = 2


def _merge_chunks(
    explore_chunks: list[RetrievedChunk], connect_chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    guaranteed_connect = connect_chunks[: min(MIN_CONNECT_SLOTS, top_k)]
    remaining_pool = explore_chunks + connect_chunks[len(guaranteed_connect) :]
    remaining_slots = top_k - len(guaranteed_connect)
    remaining = sorted(remaining_pool, key=lambda chunk: chunk.distance)[:remaining_slots]
    return guaranteed_connect + remaining


def generate_ideas(
    context: QueryContext,
    vector_store: VectorStore,
    structured_store: StructuredStore,
    llm_client: LLMClient,
    top_k: int = 5,
    include_connect_data: bool = True,
    connect_collection_name: str = CONNECT_COLLECTION,
) -> list[CreativeIdea]:
    """`include_connect_data=False` forces an Explore-only (generic) answer - used by the
    UI's opt-in "generic comparison" so it genuinely excludes loaded data, not just structured data.

    `connect_collection_name` defaults to the single shared Connect collection (Streamlit's
    usage), but callers that isolate data per session/user (e.g. the API backend) pass their
    own collection name instead.
    """
    route = decide_route(context.query, structured_data_available=structured_store.has_data())

    semantic_chunks = []
    if route.use_semantic:
        explore_chunks = vector_store.query(EXPLORE_COLLECTION, context.query, top_k=top_k)
        connect_chunks = (
            vector_store.query(connect_collection_name, context.query, top_k=top_k) if include_connect_data else []
        )
        semantic_chunks = _merge_chunks(explore_chunks, connect_chunks, top_k)

    structured_summary = structured_store.summarize_all() if route.use_structured else ""

    system_prompt = build_system_prompt(context.mode)
    user_prompt = build_user_prompt(context, semantic_chunks, structured_summary)

    raw_output = llm_client.generate(system_prompt, user_prompt)
    return _parse_ideas(raw_output)


def _strip_code_fence(text: str) -> str:
    """Strip a ```json ... ``` (or plain ``` ... ```) fence if the model wrapped its
    output in one, despite being told to respond with JSON only - a common LLM habit."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    text = text[first_newline + 1 :] if first_newline != -1 else text
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()


def _parse_ideas(raw_output: str) -> list[CreativeIdea]:
    try:
        items = json.loads(_strip_code_fence(raw_output))
        return [
            CreativeIdea(
                concept=item["concept"],
                rationale=item["rationale"],
                recommended_format=item["recommended_format"],
                source_context=item["source_context"],
                script=item.get("script"),
                grounding=item.get("grounding", "inference"),
                needs_review=item.get("needs_review", False),
                review_reason=item.get("review_reason"),
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
