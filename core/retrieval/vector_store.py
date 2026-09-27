"""Thin wrapper around a local ChromaDB instance.

Explore's general knowledge and Connect's uploaded company data are kept in
separate collections so they are never retrieved against each other.
"""
from typing import Optional

import chromadb
from chromadb.errors import NotFoundError

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

    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        max_distance: float = 1.5,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Return the closest chunks to `query_text`, dropping any that aren't actually relevant.

        `max_distance` is a relevance floor, not a tuning knob to raise casually: calibrated
        empirically (see tests/test_retrieval.py) against this project's embedding model -
        genuinely on-topic queries land ~0.9-1.3, unrelated ones land ~1.7+.

        `where` is a Chroma metadata filter (e.g. {"brand": {"$in": ["sproutmix", "general"]}})
        applied before similarity ranking, not after - so it scopes what's eligible to match at
        all, rather than just filtering the top_k results.
        """
        collection = self._collection(collection_name)
        if collection.count() == 0:
            return []

        query_kwargs = {"query_texts": [query_text], "n_results": min(top_k, collection.count())}
        if where is not None:
            query_kwargs["where"] = where
        results = collection.query(**query_kwargs)
        return [
            RetrievedChunk(text=text, source=metadata.get("source", "unknown"), distance=distance)
            for text, metadata, distance in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
            if distance <= max_distance
        ]

    def clear(self, collection_name: str) -> None:
        """Empty out a collection. A no-op (not an error) if it was never created - "clear"
        just means "ensure this is empty", and a nonexistent collection already is."""
        try:
            self._client.delete_collection(collection_name)
        except NotFoundError:
            pass

    def delete_by_source(self, collection_name: str, source: str) -> None:
        """Remove just the chunks that came from one file, identified by its `source` metadata."""
        self._collection(collection_name).delete(where={"source": source})

    def get_collection_metadata(self, collection_name: str) -> dict:
        """Collection-level metadata (not per-document) - e.g. a content hash used to detect
        whether a collection needs rebuilding. `{}` for a collection with none set, including
        one that doesn't exist yet (get_or_create_collection creates it empty)."""
        return self._collection(collection_name).metadata or {}

    def set_collection_metadata(self, collection_name: str, metadata: dict) -> None:
        """Replaces the collection's metadata entirely (not a merge)."""
        self._collection(collection_name).modify(metadata=metadata)

    def list_collection_names(self) -> list[str]:
        return [collection.name for collection in self._client.list_collections()]

    def delete_collections_with_prefix(self, prefix: str) -> None:
        """Wipe every collection whose name starts with `prefix` - used at startup to clean up
        per-session Connect collections (connect_<uuid>) left behind by sessions that ended
        without an explicit cleanup (crash, server restart, browser closed mid-session)."""
        for name in self.list_collection_names():
            if name.startswith(prefix):
                self.clear(name)
