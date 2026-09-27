"""Tests for load_explore_knowledge's hash-based rebuild: skips rebuilding an unchanged
knowledge base (so a process restart doesn't re-embed unchanged files every time), but picks
up an edited/added file on the very next call instead of silently staying stale."""
from unittest.mock import MagicMock

import pytest

from core.knowledge import loader as loader_module
from core.retrieval.vector_store import EXPLORE_COLLECTION, VectorStore


@pytest.fixture
def knowledge_dir(tmp_path, monkeypatch):
    directory = tmp_path / "explore"
    directory.mkdir()
    (directory / "general_topic.md").write_text("General D2C advice about hooks and formats.")
    (directory / "sproutmix_notes.md").write_text("SproutMix specific brand voice notes.")
    monkeypatch.setattr(loader_module, "KNOWLEDGE_DIR", directory)
    return directory


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"))


def test_first_call_populates_the_collection_and_stores_a_hash(knowledge_dir, vector_store):
    loader_module.load_explore_knowledge(vector_store)

    assert vector_store.count(EXPLORE_COLLECTION) > 0
    assert vector_store.get_collection_metadata(EXPLORE_COLLECTION).get("knowledge_hash")


def test_second_call_with_unchanged_files_is_a_no_op(knowledge_dir, vector_store, monkeypatch):
    loader_module.load_explore_knowledge(vector_store)
    original_count = vector_store.count(EXPLORE_COLLECTION)

    spy = MagicMock(wraps=vector_store.add_documents)
    monkeypatch.setattr(vector_store, "add_documents", spy)

    loader_module.load_explore_knowledge(vector_store)

    spy.assert_not_called()
    assert vector_store.count(EXPLORE_COLLECTION) == original_count


def test_editing_a_file_triggers_a_rebuild(knowledge_dir, vector_store):
    loader_module.load_explore_knowledge(vector_store)
    original_hash = vector_store.get_collection_metadata(EXPLORE_COLLECTION)["knowledge_hash"]

    (knowledge_dir / "sproutmix_notes.md").write_text("Completely different SproutMix brand voice content now.")

    loader_module.load_explore_knowledge(vector_store)

    new_hash = vector_store.get_collection_metadata(EXPLORE_COLLECTION)["knowledge_hash"]
    assert new_hash != original_hash
    results = vector_store.query(
        EXPLORE_COLLECTION, "Completely different SproutMix brand voice content", top_k=5, max_distance=999
    )
    assert any("Completely different" in r.text for r in results)


def test_adding_a_new_file_triggers_a_rebuild(knowledge_dir, vector_store):
    loader_module.load_explore_knowledge(vector_store)
    original_count = vector_store.count(EXPLORE_COLLECTION)

    (knowledge_dir / "flexwear_extra.md").write_text("Brand new FlexWear content that did not exist before.")

    loader_module.load_explore_knowledge(vector_store)

    assert vector_store.count(EXPLORE_COLLECTION) > original_count


def test_compliance_only_files_are_excluded_from_the_hash_and_dont_trigger_a_rebuild(knowledge_dir, vector_store):
    loader_module.load_explore_knowledge(vector_store)
    original_hash = vector_store.get_collection_metadata(EXPLORE_COLLECTION)["knowledge_hash"]
    original_count = vector_store.count(EXPLORE_COLLECTION)

    (knowledge_dir / "sproutmix_compliance.md").write_text("Some compliance rule text.")

    loader_module.load_explore_knowledge(vector_store)

    assert vector_store.get_collection_metadata(EXPLORE_COLLECTION)["knowledge_hash"] == original_hash
    assert vector_store.count(EXPLORE_COLLECTION) == original_count
