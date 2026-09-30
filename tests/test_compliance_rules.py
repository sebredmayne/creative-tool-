"""Regression tests for compliance rules only reaching the prompt when vector search happened
to rank them top-5. Compliance is now loaded deterministically into the system prompt, always -
these tests confirm that per-brand, independent of the query, and confirm compliance files are
excluded from the vector collection so they aren't retrieved as a second, redundant copy."""
import pytest

from core.knowledge.loader import GENERAL_COMPLIANCE_RULES, load_compliance_rules, load_explore_knowledge
from core.models import QueryContext
from core.reasoning.engine import generate_ideas
from core.reasoning.llm_client import MockLLMClient
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import EXPLORE_COLLECTION, VectorStore

# A short, distinctive substring from each brand's compliance file - unique enough that its
# presence/absence is a reliable signal, not a coincidental match.
_BRAND_MARKERS = {
    "sproutmix": "Use only under medical advice by a physician, certified dietician or nutritionist",
    "skincare": "Cosmetics claims in India sit under different rules than drugs",
    "flexwear": "exposure is body-image and greenwashing claims",
}


class _RecordingLLMClient(MockLLMClient):
    """Records the exact system_prompt it was given so tests can assert on it directly,
    instead of only being able to see MockLLMClient's fixed output."""

    def __init__(self):
        self.last_system_prompt = None

    def generate(self, system_prompt, user_prompt, response_schema=None):
        self.last_system_prompt = system_prompt
        return super().generate(system_prompt, user_prompt)


# --- load_compliance_rules(): unit-level, no vector store needed --------------------------


def test_load_compliance_rules_returns_the_right_file_per_brand():
    for brand, marker in _BRAND_MARKERS.items():
        assert marker in load_compliance_rules(brand)


def test_load_compliance_rules_falls_back_to_general_rules():
    for brand_key in (None, "general", "some_brand_with_no_compliance_file"):
        rules = load_compliance_rules(brand_key)
        assert rules == GENERAL_COMPLIANCE_RULES
        for marker in _BRAND_MARKERS.values():
            assert marker not in rules


# --- system prompt: present regardless of query, never cross-contaminated ------------------


@pytest.fixture
def llm_client():
    return _RecordingLLMClient()


@pytest.mark.parametrize("brand", ["sproutmix", "skincare", "flexwear"])
@pytest.mark.parametrize("query", ["give me a reel script", "what's a good campaign idea for the holidays"])
def test_brand_compliance_text_always_reaches_the_system_prompt(brand, query, llm_client, tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    generate_ideas(
        context=QueryContext(mode="explore", query=query),
        vector_store=vector_store,
        structured_store=StructuredStore(),
        llm_client=llm_client,
        include_connect_data=False,
        explore_brand=brand,
    )

    assert _BRAND_MARKERS[brand] in llm_client.last_system_prompt
    assert "## Non-negotiable rules" in llm_client.last_system_prompt


@pytest.mark.parametrize("brand", ["sproutmix", "skincare", "flexwear"])
def test_no_other_brands_compliance_text_appears(brand, llm_client, tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    generate_ideas(
        context=QueryContext(mode="explore", query="give me some ideas"),
        vector_store=vector_store,
        structured_store=StructuredStore(),
        llm_client=llm_client,
        include_connect_data=False,
        explore_brand=brand,
    )

    for other_brand, marker in _BRAND_MARKERS.items():
        if other_brand != brand:
            assert marker not in llm_client.last_system_prompt


def test_other_brand_gets_general_rules_only(llm_client, tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    generate_ideas(
        context=QueryContext(mode="explore", query="give me some ideas"),
        vector_store=vector_store,
        structured_store=StructuredStore(),
        llm_client=llm_client,
        include_connect_data=False,
        explore_brand="general",
    )

    assert GENERAL_COMPLIANCE_RULES in llm_client.last_system_prompt
    for marker in _BRAND_MARKERS.values():
        assert marker not in llm_client.last_system_prompt


# --- compliance files must not be duplicated into the retrieval-ranked vector collection ---


def test_compliance_files_are_excluded_from_the_explore_vector_collection(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    load_explore_knowledge(vector_store)

    # Query broadly enough (no relevance floor concerns) to see whatever's actually indexed.
    all_sources = set()
    for query in _BRAND_MARKERS.values():
        for chunk in vector_store.query(EXPLORE_COLLECTION, query, top_k=20, max_distance=999):
            all_sources.add(chunk.source)

    assert not any(source.endswith("_compliance.md") for source in all_sources)
