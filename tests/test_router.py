"""Tests for the query router's semantic/structured decision heuristics."""
from core.reasoning.router import decide_route


def test_creative_question_without_data_uses_semantic_only():
    route = decide_route("Give me 5 Instagram Reel concepts", structured_data_available=False)
    assert route.use_semantic is True
    assert route.use_structured is False


def test_numeric_question_with_data_uses_structured():
    route = decide_route("What's our average order value this quarter?", structured_data_available=True)
    assert route.use_structured is True


def test_creative_question_with_data_still_available_skips_structured():
    route = decide_route("Give me 5 creative directions based on our reviews", structured_data_available=True)
    assert route.use_structured is False


def test_structured_data_unavailable_never_triggers_structured_route():
    route = decide_route("What's our average order value?", structured_data_available=False)
    assert route.use_structured is False
