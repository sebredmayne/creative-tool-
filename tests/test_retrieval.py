"""Tests for the vector store and structured store, using synthetic data and a temp directory."""
import pandas as pd
import pytest

from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"))


def test_add_and_query_returns_relevant_chunk(vector_store):
    vector_store.add_documents(
        "test_collection",
        ids=["1", "2"],
        texts=[
            "Founder story videos build trust for new skincare brands.",
            "Retention rate dropped 4% after the pricing change in March.",
        ],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}],
    )

    results = vector_store.query("test_collection", "why did our retention fall", top_k=1)

    assert len(results) == 1
    assert "retention" in results[0].text.lower()


def test_query_on_empty_collection_returns_nothing(vector_store):
    assert vector_store.query("empty_collection", "anything", top_k=3) == []


def test_query_drops_irrelevant_results_below_relevance_floor(vector_store):
    vector_store.add_documents(
        "test_collection",
        ids=["1"],
        texts=["Founder story videos build trust for new skincare brands."],
        metadatas=[{"source": "a.md"}],
    )

    results = vector_store.query("test_collection", "what is the boiling point of water in celsius", top_k=5)

    assert results == []


def test_clear_removes_collection_contents(vector_store):
    vector_store.add_documents("test_collection", ids=["1"], texts=["some text"], metadatas=[{"source": "a"}])
    vector_store.clear("test_collection")
    assert vector_store.count("test_collection") == 0


def test_clear_on_a_never_created_collection_is_a_no_op(vector_store):
    vector_store.clear("never_created_collection")  # must not raise


def test_query_with_where_filter_only_matches_filtered_metadata(vector_store):
    # Identical text so both would tie for relevance without the filter - the assertion below
    # only holds if `where` is actually excluding the other brand, not just influencing ranking.
    vector_store.add_documents(
        "test_collection",
        ids=["1", "2"],
        texts=["Founder story videos build trust.", "Founder story videos build trust."],
        metadatas=[{"source": "a.md", "brand": "skincare"}, {"source": "b.md", "brand": "flexwear"}],
    )

    results = vector_store.query(
        "test_collection", "founder story videos", top_k=5, where={"brand": {"$in": ["skincare"]}}
    )

    assert len(results) == 1
    assert results[0].source == "a.md"


def test_list_collection_names_returns_every_collection(vector_store):
    vector_store.add_documents("collection_a", ids=["1"], texts=["x"], metadatas=[{"source": "a"}])
    vector_store.add_documents("collection_b", ids=["1"], texts=["x"], metadatas=[{"source": "a"}])

    assert set(vector_store.list_collection_names()) == {"collection_a", "collection_b"}


def test_delete_collections_with_prefix_only_removes_matching_collections(vector_store):
    vector_store.add_documents("connect_abc", ids=["1"], texts=["x"], metadatas=[{"source": "a"}])
    vector_store.add_documents("connect_def", ids=["1"], texts=["x"], metadatas=[{"source": "a"}])
    vector_store.add_documents("explore_knowledge", ids=["1"], texts=["x"], metadatas=[{"source": "a"}])

    vector_store.delete_collections_with_prefix("connect_")

    names = vector_store.list_collection_names()
    assert "connect_abc" not in names
    assert "connect_def" not in names
    assert "explore_knowledge" in names


def test_delete_by_source_only_removes_that_files_chunks(vector_store):
    vector_store.add_documents(
        "test_collection",
        ids=["1", "2"],
        texts=["chunk from file a", "chunk from file b"],
        metadatas=[{"source": "a.csv"}, {"source": "b.csv"}],
    )

    vector_store.delete_by_source("test_collection", "a.csv")

    assert vector_store.count("test_collection") == 1


def test_structured_store_summarizes_uploaded_tables():
    store = StructuredStore()
    assert store.has_data() is False

    df = pd.DataFrame({"units_sold": [120, 340, 210], "product": ["A", "B", "C"]})
    store.add("sales.csv", df)

    assert store.has_data() is True
    summary = store.summarize_all()
    assert "sales.csv" in summary
    assert "units_sold" in summary
