"""Tests for the retrieval-merge logic in the reasoning engine."""
from core.models import RetrievedChunk
from core.reasoning.engine import _merge_chunks


def _chunk(distance: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(text=f"text from {source}", source=source, distance=distance)


def test_connect_gets_reserved_slots_even_when_explore_scores_better():
    # All Explore chunks score better (lower distance) than all Connect chunks.
    explore_chunks = [_chunk(d, "explore") for d in [0.5, 0.6, 0.7, 0.8, 0.9]]
    connect_chunks = [_chunk(d, "connect") for d in [1.2, 1.3]]

    merged = _merge_chunks(explore_chunks, connect_chunks, top_k=5)

    assert len(merged) == 5
    connect_count = sum(1 for c in merged if c.source == "connect")
    assert connect_count == 2  # both Connect chunks guaranteed a slot despite scoring worse


def test_falls_back_to_pure_ranking_when_connect_is_empty():
    explore_chunks = [_chunk(d, "explore") for d in [0.5, 0.6, 0.7]]

    merged = _merge_chunks(explore_chunks, [], top_k=5)

    assert merged == explore_chunks


def test_never_exceeds_top_k():
    explore_chunks = [_chunk(d, "explore") for d in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    connect_chunks = [_chunk(d, "connect") for d in [0.4, 0.45, 0.5]]

    merged = _merge_chunks(explore_chunks, connect_chunks, top_k=5)

    assert len(merged) == 5


def test_partial_connect_results_dont_error():
    explore_chunks = [_chunk(d, "explore") for d in [0.5, 0.6, 0.7]]
    connect_chunks = [_chunk(1.0, "connect")]  # only one Connect result available

    merged = _merge_chunks(explore_chunks, connect_chunks, top_k=5)

    assert len(merged) == 4
    assert sum(1 for c in merged if c.source == "connect") == 1
