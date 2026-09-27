"""Regression tests for the cross-brand data leakage bug: a SproutMix query returning
GlowLabs data. Covers the three mechanisms that caused it - a shared Explore collection with
no brand scoping, brand-switch not clearing Connect data, and Connect data being shared across
sessions instead of per-session - using the real knowledge base and real engine entry point,
not mocks, so the fix is verified the way the bug was actually observed."""
from core.knowledge.loader import load_explore_knowledge
from core.models import QueryContext
from core.reasoning.engine import generate_ideas
from core.reasoning.llm_client import MockLLMClient
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import VectorStore


class _RecordingLLMClient(MockLLMClient):
    """Records the user_prompt it was given (which includes the retrieved knowledge chunks)
    so tests can inspect exactly what the engine actually retrieved, instead of only being
    able to see MockLLMClient's fixed output."""

    def __init__(self):
        self.last_user_prompt = None

    def generate(self, system_prompt, user_prompt):
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
