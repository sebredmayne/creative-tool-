"""Tests for the retrieval-merge logic and output parsing in the reasoning engine."""
from core.models import QueryContext, RetrievedChunk
from core.reasoning.engine import _merge_chunks, _needs_previous_query_context, _parse_ideas, generate_ideas
from core.reasoning.llm_client import MockLLMClient
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import EXPLORE_COLLECTION, VectorStore


def _chunk(distance: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(text=f"text from {source}", source=source, distance=distance)


def test_connect_gets_reserved_slots_even_when_explore_scores_better():
    # All Explore chunks score better (lower distance) than all Connect chunks.
    explore_chunks = [_chunk(d, "explore") for d in [0.5, 0.6, 0.7, 0.8, 0.9]]
    connect_chunks = [_chunk(d, "connect") for d in [1.2, 1.3]]

    merged = _merge_chunks(explore_chunks, connect_chunks, top_k=5)

    assert len(merged) == 5
    connect_count = sum(1 for c in merged if c.source == "connect")
    assert connect_count == 2  # both Connect chunks guaranteed a slot despite scoring worse


def test_falls_back_to_pure_ranking_when_connect_is_empty():
    explore_chunks = [_chunk(d, "explore") for d in [0.5, 0.6, 0.7]]

    merged = _merge_chunks(explore_chunks, [], top_k=5)

    assert merged == explore_chunks


def test_never_exceeds_top_k():
    explore_chunks = [_chunk(d, "explore") for d in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    connect_chunks = [_chunk(d, "connect") for d in [0.4, 0.45, 0.5]]

    merged = _merge_chunks(explore_chunks, connect_chunks, top_k=5)

    assert len(merged) == 5


def test_partial_connect_results_dont_error():
    explore_chunks = [_chunk(d, "explore") for d in [0.5, 0.6, 0.7]]
    connect_chunks = [_chunk(1.0, "connect")]  # only one Connect result available

    merged = _merge_chunks(explore_chunks, connect_chunks, top_k=5)

    assert len(merged) == 4
    assert sum(1 for c in merged if c.source == "connect") == 1


_VALID_JSON = """[{"concept": "A", "rationale": "r", "recommended_format": "Reel", "source_context": "s", "script": null}]"""


def test_parses_plain_json():
    ideas = _parse_ideas(_VALID_JSON)
    assert len(ideas) == 1
    assert ideas[0].concept == "A"


def test_parses_json_wrapped_in_markdown_code_fence():
    fenced = f"```json\n{_VALID_JSON}\n```"
    ideas = _parse_ideas(fenced)
    assert len(ideas) == 1
    assert ideas[0].concept == "A"


def test_parses_json_wrapped_in_plain_code_fence():
    fenced = f"```\n{_VALID_JSON}\n```"
    ideas = _parse_ideas(fenced)
    assert len(ideas) == 1


def test_falls_back_gracefully_on_genuinely_invalid_output():
    ideas = _parse_ideas("this is not json at all")
    assert len(ideas) == 1
    assert ideas[0].concept == "Unparsed model output"


# --- _needs_previous_query_context: heuristic for bare follow-ups -------------------------


def test_short_query_needs_previous_context():
    assert _needs_previous_query_context("Hindi versions") is True


def test_query_starting_with_make_needs_previous_context():
    assert _needs_previous_query_context("Make idea 2 funnier and punchier for a younger audience") is True


def test_query_starting_with_now_needs_previous_context():
    assert _needs_previous_query_context("Now turn that into a static ad instead of a reel") is True


def test_give_me_versions_query_needs_previous_context():
    assert _needs_previous_query_context("Give me Hindi versions of these same ideas please") is True


def test_long_self_contained_query_does_not_need_previous_context():
    assert (
        _needs_previous_query_context(
            "Give me 3 Reel scripts about why kids reject milk mix and how we solve it"
        )
        is False
    )


def test_empty_query_does_not_need_previous_context():
    assert _needs_previous_query_context("   ") is False


# --- generate_ideas: previous_query actually reaches retrieval ----------------------------


class _RecordingLLMClient(MockLLMClient):
    def __init__(self):
        self.last_user_prompt = None

    def generate(self, system_prompt, user_prompt):
        self.last_user_prompt = user_prompt
        return super().generate(system_prompt, user_prompt)


def test_bare_followup_only_retrieves_relevant_context_when_previous_query_is_supplied(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    vector_store.add_documents(
        "test_connect",
        ids=["1"],
        texts=["Reviewers repeatedly complain the milk mix leaves a chalky sediment at the bottom of the glass."],
        metadatas=[{"source": "reviews.csv"}],
    )

    llm_client = _RecordingLLMClient()
    generate_ideas(
        context=QueryContext(mode="connect", query="make it funnier"),
        vector_store=vector_store,
        structured_store=StructuredStore(),
        llm_client=llm_client,
        connect_collection_name="test_connect",
        previous_query="why do reviewers complain about chalky sediment in the milk mix",
    )
    assert "chalky sediment" in llm_client.last_user_prompt.lower()

    llm_client_without_context = _RecordingLLMClient()
    generate_ideas(
        context=QueryContext(mode="connect", query="make it funnier"),
        vector_store=vector_store,
        structured_store=StructuredStore(),
        llm_client=llm_client_without_context,
        connect_collection_name="test_connect",
        previous_query=None,
    )
    assert "chalky sediment" not in llm_client_without_context.last_user_prompt.lower()
