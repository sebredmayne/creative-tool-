"""Tests for the eval harness's own checking logic - check_idea, check_implied_claims, and
judge_idea - using a fake/mocked LLM client so these run with zero real API calls, even though
`python -m evals.run` itself is deliberately excluded from pytest."""
from core.models import CreativeIdea
from evals.run import check_idea, check_implied_claims, judge_idea


def _idea(**overrides) -> CreativeIdea:
    defaults = dict(
        concept="Morning Routine Reel",
        rationale="Parents relate to the morning rush.",
        recommended_format="Instagram Reel",
        source_context="sproutmix_market_and_personas.md",
        script="Beat 1: ...",
        grounding="inference",
        needs_review=False,
        review_reason=None,
    )
    defaults.update(overrides)
    return CreativeIdea(**defaults)


# --- check_idea: absolute vs conditional banned phrases ------------------------------------


def test_absolute_phrase_fails_even_when_flagged_for_review():
    idea = _idea(rationale="This claims the product boosts immunity.", needs_review=True)
    failures, warnings = check_idea(idea, "sproutmix", banned_phrases=["boosts immunity"], phrases_are_absolute=True)
    assert any("boosts immunity" in f for f in failures)
    assert warnings == []


def test_absolute_phrase_fails_and_notes_missing_review_when_not_flagged():
    idea = _idea(rationale="This claims the product boosts immunity.", needs_review=False)
    failures, _ = check_idea(idea, "sproutmix", banned_phrases=["boosts immunity"], phrases_are_absolute=True)
    assert any("needs_review is False" in f for f in failures)


def test_conditional_phrase_with_needs_review_is_a_warning_not_a_failure():
    idea = _idea(script="These leggings are true to size for most reviewers.", needs_review=True)
    failures, warnings = check_idea(idea, "flexwear", banned_phrases=["true to size"], phrases_are_absolute=False)
    assert failures == []
    assert any("true to size" in w for w in warnings)


def test_conditional_phrase_without_needs_review_is_a_failure():
    idea = _idea(script="These leggings are true to size for most reviewers.", needs_review=False)
    failures, warnings = check_idea(idea, "flexwear", banned_phrases=["true to size"], phrases_are_absolute=False)
    assert any("true to size" in f for f in failures)
    assert warnings == []


def test_no_banned_phrase_present_is_clean():
    idea = _idea(script="Comfortable everyday leggings for any workout.")
    failures, warnings = check_idea(idea, "flexwear", banned_phrases=["true to size"], phrases_are_absolute=False)
    assert failures == [] and warnings == []


# --- check_idea: brand leakage + structure --------------------------------------------------


def test_flags_another_brands_name():
    idea = _idea(rationale="Unlike GlowLabs, our product is different.")
    failures, _ = check_idea(idea, "sproutmix", banned_phrases=[], phrases_are_absolute=True)
    assert any("GlowLabs" in f for f in failures)


def test_flags_unparsed_model_output_as_invalid_structure():
    idea = _idea(concept="Unparsed model output", recommended_format="n/a")
    failures, warnings = check_idea(idea, "sproutmix", banned_phrases=[], phrases_are_absolute=True)
    assert any("unparsed model output" in f for f in failures)
    assert warnings == []


# --- check_implied_claims --------------------------------------------------------------------


def test_check_implied_claims_detects_a_keyword_match():
    idea = _idea(script="After a week, kids seem calmer at bedtime.")
    failures = check_implied_claims(idea, ["mood"])
    assert any("mood" in f for f in failures)


def test_check_implied_claims_no_match_is_clean():
    idea = _idea(script="A simple, tasty milk mix for breakfast.")
    failures = check_implied_claims(idea, ["mood", "behaviour", "growth", "immunity", "brain"])
    assert failures == []


def test_check_implied_claims_only_checks_requested_categories():
    idea = _idea(script="Great for a growing kid's daily routine.")  # "grow" would match "growth"
    failures = check_implied_claims(idea, ["mood"])  # but only "mood" was requested
    assert failures == []


# --- judge_idea: uses a fake LLM client, never a real API call ------------------------------


class _FakeJudgeClient:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def generate(self, system_prompt, user_prompt):
        return self._response_text


def test_judge_idea_returns_no_failures_when_compliant():
    client = _FakeJudgeClient('{"compliant": true, "issues": []}')
    failures = judge_idea(_idea(), "sproutmix", client)
    assert failures == []


def test_judge_idea_returns_failures_when_not_compliant():
    client = _FakeJudgeClient('{"compliant": false, "issues": ["implies a growth outcome"]}')
    failures = judge_idea(_idea(), "sproutmix", client)
    assert any("implies a growth outcome" in f for f in failures)


def test_judge_idea_handles_markdown_fenced_response():
    client = _FakeJudgeClient('```json\n{"compliant": false, "issues": ["bad claim"]}\n```')
    failures = judge_idea(_idea(), "sproutmix", client)
    assert any("bad claim" in f for f in failures)


def test_judge_idea_handles_unparseable_response_without_raising():
    client = _FakeJudgeClient("not json at all")
    failures = judge_idea(_idea(), "sproutmix", client)
    assert len(failures) == 1
    assert "LLM judge" in failures[0]
