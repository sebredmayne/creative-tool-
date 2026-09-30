"""Orchestrates one creative-generation request: route -> retrieve -> prompt -> generate -> parse.

Always draws on the curated Explore knowledge base, and additionally on
whatever's been loaded into Connect (if anything). The two are merged
rather than mutually exclusive - a question about one category shouldn't
lose its curated knowledge just because unrelated data from a different
category happens to be loaded.
"""
import json
from typing import Optional

from core.knowledge.loader import load_compliance_rules
from core.models import CreativeIdea, QueryContext
from core.reasoning.llm_client import CREATIVE_IDEA_SCHEMA, LLMClient
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

# Below this word count, a query is treated as too short to carry its own retrieval context.
_SHORT_QUERY_WORD_LIMIT = 8


def _needs_previous_query_context(query: str) -> bool:
    """A short follow-up like "Hindi versions of these" or "make it funnier" carries almost no
    retrieval signal on its own - what "these"/"it" refers to is the previous turn's query, not
    anything in this one. Heuristic, not exhaustive: under the word limit, or opening with a
    word that's characteristic of a bare follow-up ("make ...", "now ...", "give me ...
    versions")."""
    stripped = query.strip().lower()
    if not stripped:
        return False
    if len(stripped.split()) < _SHORT_QUERY_WORD_LIMIT:
        return True
    if stripped.startswith("make") or stripped.startswith("now"):
        return True
    if stripped.startswith("give me") and "version" in stripped:
        return True
    return False


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
    explore_brand: Optional[str] = None,
    previous_query: Optional[str] = None,
    previous_ideas: Optional[list[CreativeIdea]] = None,
) -> list[CreativeIdea]:
    """`include_connect_data=False` forces an Explore-only (generic) answer - used by the
    UI's opt-in "generic comparison" so it genuinely excludes loaded data, not just structured data.

    `connect_collection_name` defaults to the single shared Connect collection, but every real
    session passes its own `connect_<uuid>` collection instead (see core.session.CopilotSession)
    so one session's uploads are never visible to another's queries - the shared default only
    matters for callers (tests) that don't need that isolation.

    `explore_brand` scopes the Explore query to one brand's own knowledge plus brand-agnostic
    "general" knowledge, so a query about one brand can never retrieve another brand's chunks
    (e.g. a SproutMix query pulling in GlowLabs data). It also selects which brand's compliance
    rules get loaded deterministically into the system prompt (see load_compliance_rules) -
    those are non-negotiable and always included regardless of retrieval, so this is not only a
    retrieval-scoping knob. Pass the brand key itself (e.g. "sproutmix"), or "general" for a
    brand-less session ("Other brand" in the UI). Left as None (the default), the Explore query
    is unscoped and the general compliance rules are used - only intended for callers that
    don't have a brand concept at all.

    `previous_query` and `previous_ideas` carry context from the prior turn, for follow-ups
    that reach here with no context of their own: `previous_query` gets appended to the
    *retrieval* text (not the prompt's "User request") for short/bare follow-ups (see
    _needs_previous_query_context) so retrieval isn't done on a query like "now make it a
    static ad" in isolation; `previous_ideas` is passed straight through to build_user_prompt
    so the model can see what "idea 2" or "these" actually refers to.
    """
    route = decide_route(context.query, structured_data_available=structured_store.has_data())

    explore_where = None
    if explore_brand is not None:
        brands = [explore_brand] if explore_brand == "general" else [explore_brand, "general"]
        explore_where = {"brand": {"$in": brands}}

    retrieval_query = context.query
    if previous_query and _needs_previous_query_context(context.query):
        retrieval_query = f"{previous_query} {context.query}"

    semantic_chunks = []
    if route.use_semantic:
        explore_chunks = vector_store.query(EXPLORE_COLLECTION, retrieval_query, top_k=top_k, where=explore_where)
        connect_chunks = (
            vector_store.query(connect_collection_name, retrieval_query, top_k=top_k)
            if include_connect_data
            else []
        )
        semantic_chunks = _merge_chunks(explore_chunks, connect_chunks, top_k)

    structured_summary = structured_store.summarize_all() if route.use_structured else ""

    compliance_rules = load_compliance_rules(explore_brand)
    system_prompt = build_system_prompt(context.mode, compliance_rules)
    user_prompt = build_user_prompt(context, semantic_chunks, structured_summary, previous_ideas=previous_ideas)

    raw_output = llm_client.generate(system_prompt, user_prompt, response_schema=CREATIVE_IDEA_SCHEMA)
    return _parse_ideas(raw_output)


def strip_code_fence(text: str) -> str:
    """Strip a ```json ... ``` (or plain ``` ... ```) fence if the model wrapped its
    output in one, despite being told to respond with JSON only - a common LLM habit.

    Public (not `_`-prefixed): also reused by evals/run.py's --judge mode, which parses a
    similarly-shaped JSON response from its own separate LLM call.
    """
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
        items = json.loads(strip_code_fence(raw_output))
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
