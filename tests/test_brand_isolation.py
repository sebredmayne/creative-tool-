"""Regression tests for the cross-brand data leakage bug: a SproutMix query returning
GlowLabs data. Covers the three mechanisms that caused it - a shared Explore collection with
no brand scoping, brand-switch not clearing Connect data, and Connect data being shared across
sessions instead of per-session - using the real knowledge base and real engine entry point,
not mocks, so the fix is verified the way the bug was actually observed."""
import pytest

from core import custom_brands
from core.knowledge.loader import load_explore_knowledge
from core.models import BrandProfile, QueryContext
from core.reasoning.engine import generate_ideas
from core.reasoning.llm_client import MockLLMClient
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import EXPLORE_COLLECTION, VectorStore


class _RecordingLLMClient(MockLLMClient):
    """Records the user_prompt it was given (which includes the retrieved knowledge chunks)
    so tests can inspect exactly what the engine actually retrieved, instead of only being
    able to see MockLLMClient's fixed output."""

    def __init__(self):
        self.last_user_prompt = None

    def generate(self, system_prompt, user_prompt, response_schema=None):
        self.last_user_prompt = user_prompt
        return super().generate(system_prompt, user_prompt)


def _engine_setup(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    load_explore_knowledge(vector_store)
    llm_client = _RecordingLLMClient()
    return vector_store, llm_client


def test_sproutmix_query_never_retrieves_flexwear_or_skincare_explore_chunks(tmp_path):
    vector_store, llm_client = _engine_setup(tmp_path)

    generate_ideas(
        context=QueryContext(mode="explore", query="why do kids reject milk mix"),
        vector_store=vector_store,
        structured_store=StructuredStore(),
        llm_client=llm_client,
        include_connect_data=False,
        explore_brand="sproutmix",
    )

    assert "flexwear" not in llm_client.last_user_prompt.lower()
    assert "skincare" not in llm_client.last_user_prompt.lower()


def test_other_brand_query_only_retrieves_general_explore_chunks(tmp_path):
    vector_store, llm_client = _engine_setup(tmp_path)

    generate_ideas(
        context=QueryContext(mode="explore", query="what makes a good D2C hook"),
        vector_store=vector_store,
        structured_store=StructuredStore(),
        llm_client=llm_client,
        include_connect_data=False,
        explore_brand="general",
    )

    for brand_word in ("sproutmix", "flexwear", "skincare"):
        assert brand_word not in llm_client.last_user_prompt.lower()


def test_switching_brands_leaves_zero_chunks_from_the_old_brand(tmp_path):
    """Mirrors what app.py's load_brand_preset does on a brand switch: clear the session's
    Connect collection, then load the new brand's data into it."""
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    session_collection = "connect_test-session"

    vector_store.add_documents(
        session_collection,
        ids=["1"],
        texts=["SproutMix parents complain the milk mix is too sweet for younger kids."],
        metadatas=[{"source": "[sample] sproutmix_reviews.csv"}],
    )

    # Simulate switching brands: clear, then load the new brand's data.
    vector_store.clear(session_collection)
    vector_store.add_documents(
        session_collection,
        ids=["1"],
        texts=["GlowLabs customers love how the vitamin C face wash brightens skin over time."],
        metadatas=[{"source": "[sample] skincare_reviews.csv"}],
    )

    results = vector_store.query(session_collection, "milk mix sweetness complaints", top_k=5)
    assert all("sproutmix" not in r.source.lower() for r in results)
    assert vector_store.count(session_collection) == 1


def test_two_sessions_never_see_each_others_uploads(tmp_path):
    """Two Streamlit sessions share one VectorStore instance (it's @st.cache_resource-cached),
    but each gets its own connect_<uuid> collection - this is the actual isolation mechanism."""
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    session_a = "connect_session-a"
    session_b = "connect_session-b"

    vector_store.add_documents(
        session_a,
        ids=["1"],
        texts=["Session A uploaded its own private sales data."],
        metadatas=[{"source": "session_a_upload.csv"}],
    )
    vector_store.add_documents(
        session_b,
        ids=["1"],
        texts=["Session B uploaded a completely different private dataset."],
        metadatas=[{"source": "session_b_upload.csv"}],
    )

    results_for_a = vector_store.query(session_a, "private data", top_k=5)
    results_for_b = vector_store.query(session_b, "private data", top_k=5)

    assert all(r.source == "session_a_upload.csv" for r in results_for_a)
    assert all(r.source == "session_b_upload.csv" for r in results_for_b)


@pytest.fixture
def temp_brands_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_brands, "BRANDS_DIR", tmp_path / "brands")


def _save_custom_brand(slug: str, guide_text: str) -> None:
    profile = BrandProfile(
        slug=slug,
        label=slug,
        brand_name=slug,
        description="",
        context={"brand": slug, "product": "", "customer": "", "category": "", "objective": ""},
        compliance_rules=f"Never make unverified claims for {slug}.",
        guide_filename="guide.md",
        is_custom=True,
    )
    custom_brands.save_custom_brand(profile, guide_text=guide_text, guide_filename="guide.md", guide_bytes=b"", data_files=[])


def test_custom_brands_explore_chunks_never_cross_contaminate(temp_brands_dir, tmp_path):
    """Two custom brands, each with distinctive guide content - neither's chunks should ever
    be retrievable when scoped to the other, or to a preset."""
    _save_custom_brand("joes-coffee", "Joe's Coffee Co. sources single-origin arabica beans from Ethiopia and Colombia.")
    _save_custom_brand("mias-candles", "Mia's Candles hand-pours soy wax candles scented with lavender and cedarwood.")

    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    load_explore_knowledge(vector_store)

    coffee_chunks = vector_store.query(
        EXPLORE_COLLECTION, "arabica beans Ethiopia Colombia", top_k=10, max_distance=999,
        where={"brand": {"$in": ["joes-coffee", "general"]}},
    )
    assert coffee_chunks  # sanity: it was actually indexed
    assert all(c.source != "mias-candles_brand_guide" for c in coffee_chunks)

    candle_chunks = vector_store.query(
        EXPLORE_COLLECTION, "soy wax lavender cedarwood candles", top_k=10, max_distance=999,
        where={"brand": {"$in": ["mias-candles", "general"]}},
    )
    assert candle_chunks
    assert all(c.source != "joes-coffee_brand_guide" for c in candle_chunks)

    # Scoped to a preset, neither custom brand's guide should be retrievable either.
    sproutmix_chunks = vector_store.query(
        EXPLORE_COLLECTION, "arabica beans soy wax candles", top_k=20, max_distance=999,
        where={"brand": {"$in": ["sproutmix", "general"]}},
    )
    assert all(c.source not in ("joes-coffee_brand_guide", "mias-candles_brand_guide") for c in sproutmix_chunks)


def test_deleting_a_custom_brand_removes_its_chunks_on_next_rebuild(temp_brands_dir, tmp_path):
    _save_custom_brand("joes-coffee", "Joe's Coffee Co. sources single-origin arabica beans from Ethiopia.")

    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    load_explore_knowledge(vector_store)
    assert vector_store.query(EXPLORE_COLLECTION, "arabica beans", top_k=5, max_distance=999)

    custom_brands.delete_custom_brand("joes-coffee")
    load_explore_knowledge(vector_store)  # hash changed (brand gone) - triggers a rebuild

    remaining = vector_store.query(EXPLORE_COLLECTION, "arabica beans", top_k=5, max_distance=999)
    assert all(c.source != "joes-coffee_brand_guide" for c in remaining)
