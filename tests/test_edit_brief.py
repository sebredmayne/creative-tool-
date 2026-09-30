"""Tests for generate_edit_brief: response parsing/schema validation, contiguous-timestamp and
duration validation, and that compliance notes are carried through from the source idea - not
re-derived by the edit-brief LLM call. Uses a fake LLM client, no real API calls."""
import json

import pytest

from core.models import BriefScene, CreativeIdea, EditBrief
from core.reasoning.edit_brief import (
    EditBriefError,
    edit_brief_to_json,
    edit_brief_to_markdown,
    generate_edit_brief,
    validate_edit_brief,
)


class _FakeClient:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_response_schema = None

    def generate(self, system_prompt, user_prompt, response_schema=None):
        self.last_response_schema = response_schema
        return self._response_text


def _idea(**overrides) -> CreativeIdea:
    defaults = dict(
        concept="Morning Chalky Texture Fix",
        rationale="Addresses the #1 review complaint directly.",
        recommended_format="Instagram Reel",
        source_context="sproutmix_reviews.csv",
        script="0:00-0:05 hook, 0:05-0:30 body",
        grounding="brand_data",
        needs_review=False,
        review_reason=None,
    )
    defaults.update(overrides)
    return CreativeIdea(**defaults)


def _valid_response(duration=30) -> str:
    return json.dumps(
        {
            "aspect_ratio": "9:16",
            "duration_seconds": duration,
            "alternative_hooks": ["Hook A", "Hook B"],
            "scenes": [
                {
                    "timestamp_range": "0:00-0:05",
                    "beat_type": "hook",
                    "voiceover_or_onscreen_text": "Ever notice a chalky layer at the bottom?",
                    "visual_description": "Close-up of a glass with sediment.",
                    "b_roll_suggestions": ["Pouring milk mix", "Stirring"],
                    "caption_text": "Wait for it...",
                    "footage_note": None,
                },
                {
                    "timestamp_range": "0:05-0:25",
                    "beat_type": "body",
                    "voiceover_or_onscreen_text": "Here's why - and how we fixed it.",
                    "visual_description": "Split screen: old formula vs new.",
                    "b_roll_suggestions": [],
                    "caption_text": "",
                    "footage_note": "needs UGC selfie shot",
                },
                {
                    "timestamp_range": "0:25-0:30",
                    "beat_type": "cta",
                    "voiceover_or_onscreen_text": "Try the new SproutMix today.",
                    "visual_description": "Product shot with logo.",
                    "b_roll_suggestions": [],
                    "caption_text": "Link in bio",
                    "footage_note": None,
                },
            ],
            "music_mood_note": "Upbeat, playful, kid-friendly.",
        }
    )


# --- generate_edit_brief: parsing / schema validation --------------------------------------


def test_generate_edit_brief_parses_a_valid_response():
    client = _FakeClient(_valid_response())
    brief = generate_edit_brief(_idea(), "Sunny Sprout", "Never claim health outcomes.", client)

    assert brief.concept == "Morning Chalky Texture Fix"
    assert brief.brand == "Sunny Sprout"
    assert brief.format == "Instagram Reel"
    assert brief.aspect_ratio == "9:16"
    assert brief.duration_seconds == 30
    assert len(brief.scenes) == 3
    assert brief.scenes[0].beat_type == "hook"
    assert brief.scenes[1].footage_note == "needs UGC selfie shot"
    assert len(brief.alternative_hooks) == 2


def test_generate_edit_brief_parses_markdown_fenced_response():
    fenced = f"```json\n{_valid_response()}\n```"
    client = _FakeClient(fenced)
    brief = generate_edit_brief(_idea(), "Sunny Sprout", "rules", client)
    assert len(brief.scenes) == 3


def test_generate_edit_brief_passes_the_schema_to_the_llm_client():
    client = _FakeClient(_valid_response())
    generate_edit_brief(_idea(), "Sunny Sprout", "rules", client)
    assert client.last_response_schema is not None
    assert "scenes" in client.last_response_schema["properties"]


def test_generate_edit_brief_raises_on_invalid_json():
    client = _FakeClient("not json at all")
    with pytest.raises(EditBriefError):
        generate_edit_brief(_idea(), "Sunny Sprout", "rules", client)


def test_generate_edit_brief_raises_when_a_required_scene_field_is_missing():
    broken = json.dumps(
        {
            "aspect_ratio": "9:16",
            "duration_seconds": 30,
            "alternative_hooks": [],
            "scenes": [{"timestamp_range": "0:00-0:30", "beat_type": "hook"}],  # missing text/visual
            "music_mood_note": "",
        }
    )
    client = _FakeClient(broken)
    with pytest.raises(EditBriefError):
        generate_edit_brief(_idea(), "Sunny Sprout", "rules", client)


# --- validate_edit_brief: contiguous timestamps + duration match ---------------------------


def _scene(start, end, beat="body"):
    return BriefScene(
        timestamp_range=f"{start}-{end}",
        beat_type=beat,
        voiceover_or_onscreen_text="text",
        visual_description="visual",
    )


def _brief(scenes, duration_seconds):
    return EditBrief(concept="c", brand="b", format="f", duration_seconds=duration_seconds, scenes=scenes)


def test_validate_accepts_contiguous_scenes_matching_duration():
    scenes = [_scene("0:00", "0:05", "hook"), _scene("0:05", "0:25", "body"), _scene("0:25", "0:30", "cta")]
    validate_edit_brief(_brief(scenes, 30))  # must not raise


def test_validate_rejects_a_gap_between_scenes():
    scenes = [_scene("0:00", "0:05"), _scene("0:10", "0:30")]  # gap from 0:05 to 0:10
    with pytest.raises(EditBriefError):
        validate_edit_brief(_brief(scenes, 30))


def test_validate_rejects_an_overlap_between_scenes():
    scenes = [_scene("0:00", "0:10"), _scene("0:05", "0:30")]  # overlaps 0:05-0:10
    with pytest.raises(EditBriefError):
        validate_edit_brief(_brief(scenes, 30))


def test_validate_rejects_scenes_that_dont_start_at_zero():
    scenes = [_scene("0:05", "0:30")]
    with pytest.raises(EditBriefError):
        validate_edit_brief(_brief(scenes, 30))


def test_validate_rejects_scenes_that_dont_sum_to_the_stated_duration():
    scenes = [_scene("0:00", "0:20")]
    with pytest.raises(EditBriefError):
        validate_edit_brief(_brief(scenes, 30))  # ends at 20s, duration says 30s


def test_validate_rejects_a_scene_that_doesnt_move_forward():
    scenes = [_scene("0:00", "0:00")]
    with pytest.raises(EditBriefError):
        validate_edit_brief(_brief(scenes, 30))


def test_validate_rejects_empty_scenes():
    with pytest.raises(EditBriefError):
        validate_edit_brief(_brief([], 30))


def test_generate_edit_brief_raises_when_scenes_are_not_contiguous():
    non_contiguous = json.dumps(
        {
            "aspect_ratio": "9:16",
            "duration_seconds": 30,
            "alternative_hooks": [],
            "scenes": [
                {
                    "timestamp_range": "0:00-0:05",
                    "beat_type": "hook",
                    "voiceover_or_onscreen_text": "x",
                    "visual_description": "y",
                },
                {
                    "timestamp_range": "0:10-0:30",  # gap
                    "beat_type": "body",
                    "voiceover_or_onscreen_text": "x",
                    "visual_description": "y",
                },
            ],
            "music_mood_note": "",
        }
    )
    client = _FakeClient(non_contiguous)
    with pytest.raises(EditBriefError):
        generate_edit_brief(_idea(), "Sunny Sprout", "rules", client)


# --- compliance notes are carried through from the idea, not re-derived -------------------


def test_compliance_notes_carried_through_when_idea_needs_review():
    idea = _idea(needs_review=True, review_reason="Implies an immunity outcome.")
    client = _FakeClient(_valid_response())
    brief = generate_edit_brief(idea, "Sunny Sprout", "rules", client)
    assert brief.compliance_notes == "Implies an immunity outcome."


def test_compliance_notes_is_none_when_idea_does_not_need_review():
    idea = _idea(needs_review=False, review_reason=None)
    client = _FakeClient(_valid_response())
    brief = generate_edit_brief(idea, "Sunny Sprout", "rules", client)
    assert brief.compliance_notes is None


def test_compliance_notes_ignores_any_llm_provided_compliance_field():
    """Even if the model's JSON somehow included its own compliance opinion, it has no schema
    field for one - compliance_notes must come only from the idea's own review_reason."""
    idea = _idea(needs_review=True, review_reason="Idea-level reason.")
    response = json.loads(_valid_response())
    response["compliance_notes"] = "Model's own opinion - should be ignored entirely."
    client = _FakeClient(json.dumps(response))
    brief = generate_edit_brief(idea, "Sunny Sprout", "rules", client)
    assert brief.compliance_notes == "Idea-level reason."


# --- output rendering: JSON / Markdown -----------------------------------------------------


def test_edit_brief_to_json_round_trips_the_key_fields():
    client = _FakeClient(_valid_response())
    brief = generate_edit_brief(_idea(), "Sunny Sprout", "rules", client)

    parsed = json.loads(edit_brief_to_json(brief))

    assert parsed["concept"] == brief.concept
    assert len(parsed["scenes"]) == 3
    assert parsed["scenes"][0]["timestamp_range"] == "0:00-0:05"


def test_edit_brief_to_markdown_includes_scenes_and_compliance_note():
    idea = _idea(needs_review=True, review_reason="Flag this before publishing.")
    client = _FakeClient(_valid_response())
    brief = generate_edit_brief(idea, "Sunny Sprout", "rules", client)

    markdown = edit_brief_to_markdown(brief)

    assert "Flag this before publishing." in markdown
    assert "0:00-0:05" in markdown
    assert "Ever notice a chalky layer" in markdown
