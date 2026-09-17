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

    def query(self, collection_name: str, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        collection = self._collection(collection_name)
        if collection.count() == 0:
            return []

        results = collection.query(query_texts=[query_text], n_results=min(top_k, collection.count()))
        return [
            RetrievedChunk(text=text, source=metadata.get("source", "unknown"), distance=distance)
            for text, metadata, distance in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]

    def clear(self, collection_name: str) -> None:
        self._client.delete_collection(collection_name)
