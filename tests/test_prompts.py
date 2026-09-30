"""Tests for build_user_prompt's previous_ideas section - the fix for follow-ups like "Hindi
versions of these" reaching the model with no idea what "these" refers to."""
from core.knowledge.loader import load_compliance_rules
from core.models import CreativeIdea, QueryContext
from core.reasoning.prompts import OUTPUT_INSTRUCTIONS, build_user_prompt

_CONTEXT = QueryContext(mode="explore", query="give me a reel script")


def _idea(concept="Milk mix morning routine", script="Beat 1: ...", fmt="Instagram Reel"):
    return CreativeIdea(
        concept=concept,
        rationale="Parents relate to the morning rush.",
        recommended_format=fmt,
        source_context="sproutmix_market_and_personas.md",
        script=script,
    )


def test_first_message_has_no_previous_ideas_section():
    prompt = build_user_prompt(_CONTEXT, semantic_chunks=[], structured_summary="", previous_ideas=None)
    assert "previous turn" not in prompt.lower()


def test_empty_previous_ideas_list_also_omits_the_section():
    prompt = build_user_prompt(_CONTEXT, semantic_chunks=[], structured_summary="", previous_ideas=[])
    assert "previous turn" not in prompt.lower()


def test_followup_prompt_includes_previous_ideas():
    previous = [_idea(concept="Milk mix morning routine"), _idea(concept="Chalky sediment doubt")]
    prompt = build_user_prompt(_CONTEXT, semantic_chunks=[], structured_summary="", previous_ideas=previous)

    assert "previous turn" in prompt.lower()
    assert "Milk mix morning routine" in prompt
    assert "Chalky sediment doubt" in prompt
    assert "Instagram Reel" in prompt


def test_previous_idea_script_is_truncated_not_dropped_when_long():
    long_script = "x" * 1000
    previous = [_idea(script=long_script)]
    prompt = build_user_prompt(_CONTEXT, semantic_chunks=[], structured_summary="", previous_ideas=previous)

    assert long_script not in prompt  # full script would bloat every subsequent prompt
    assert "x" * 50 in prompt  # but a meaningful prefix is still there, not dropped entirely


# --- regression guard: every output field must stay clean of implied-outcome language, not --
# --- just "script" - found via evals/run.py's forbidden_implied_claims case flagging the ----
# --- model's own *rationale* ("...instead of growth outcomes") and *concept* ("...Protein ---
# --- Focus"), never the visible ad copy. Fixed at the prompt level (OUTPUT_INSTRUCTIONS + ---
# --- sproutmix_compliance.md), deliberately not by narrowing what evals/run.py checks. ------


def test_output_instructions_warn_against_implied_claims_in_every_field():
    lowered = OUTPUT_INSTRUCTIONS.lower()
    assert "concept" in lowered and "rationale" in lowered  # names the non-script fields
    assert "growth" in lowered and "focus" in lowered  # names concrete trigger words, not vaguely


def test_sproutmix_compliance_warns_against_echoing_avoided_words_back():
    rules = load_compliance_rules("sproutmix")
    assert "avoids growth outcomes" in rules.lower()  # the exact anti-pattern this guards against
