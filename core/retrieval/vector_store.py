"""Thin wrapper around a local ChromaDB instance.

Explore's general knowledge and Connect's uploaded company data are kept in
separate collections so they are never retrieved against each other.
"""
import chromadb

from core.models import RetrievedChunk

EXPLORE_COLLECTION = "explore_knowledge"
CONNECT_COLLECTION = "connect_knowledge"


class VectorStore:
    def __init__(self, persist_dir: str = "data/chroma"):
        self._client = chromadb.PersistentClient(path=persist_dir)

    def _collection(self, name: str):
        return self._client.get_or_create_collection(name)

    def count(self, collection_name: str) -> int:
        return self._collection(collection_name).count()

    def add_documents(self, collection_name: str, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        if not texts:
            return
        self._collection(collection_name).add(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, collection_name: str, query_text: str, top_k: int = 5, max_distance: float = 1.5) -> list[RetrievedChunk]:
        """Return the closest chunks to `query_text`, dropping any that aren't actually relevant.

        `max_distance` is a relevance floor, not a tuning knob to raise casually: calibrated
        empirically (see tests/test_retrieval.py) against this project's embedding model -
        genuinely on-topic queries land ~0.9-1.3, unrelated ones land ~1.7+.
        """
        collection = self._collection(collection_name)
        if collection.count() == 0:
            return []

        results = collection.query(query_texts=[query_text], n_results=min(top_k, collection.count()))
        return [
            RetrievedChunk(text=text, source=metadata.get("source", "unknown"), distance=distance)
            for text, metadata, distance in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
            if distance <= max_distance
        ]

    def clear(self, collection_name: str) -> None:
        self._client.delete_collection(collection_name)

    def delete_by_source(self, collection_name: str, source: str) -> None:
        """Remove just the chunks that came from one file, identified by its `source` metadata."""
        self._collection(collection_name).delete(where={"source": source})
