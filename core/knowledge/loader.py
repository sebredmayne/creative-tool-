"""Loads the curated Explore-mode D2C knowledge base (markdown files) into the vector store."""
from pathlib import Path

from core.ingestion.chunking import chunk_text
from core.retrieval.vector_store import EXPLORE_COLLECTION, VectorStore

KNOWLEDGE_DIR = Path(__file__).parent / "explore"


def load_explore_knowledge(vector_store: VectorStore) -> None:
    """Populate the Explore collection from local markdown files, if it's empty.

    Safe to call on every app startup - it's a no-op once the collection is populated.
    """
    if vector_store.count(EXPLORE_COLLECTION) > 0:
        return

    ids, texts, metadatas = [], [], []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(content, chunk_size=150, overlap=30)):
            ids.append(f"{path.stem}-{i}")
            texts.append(chunk)
            metadatas.append({"source": path.name})

    vector_store.add_documents(EXPLORE_COLLECTION, ids=ids, texts=texts, metadatas=metadatas)
