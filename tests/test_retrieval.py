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


def test_clear_removes_collection_contents(vector_store):
    vector_store.add_documents("test_collection", ids=["1"], texts=["some text"], metadatas=[{"source": "a"}])
    vector_store.clear("test_collection")
    assert vector_store.count("test_collection") == 0


def test_structured_store_summarizes_uploaded_tables():
    store = StructuredStore()
    assert store.has_data() is False

    df = pd.DataFrame({"units_sold": [120, 340, 210], "product": ["A", "B", "C"]})
    store.add("sales.csv", df)

    assert store.has_data() is True
    summary = store.summarize_all()
    assert "sales.csv" in summary
    assert "units_sold" in summary
