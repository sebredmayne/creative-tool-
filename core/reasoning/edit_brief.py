"""Turns an already-generated CreativeIdea into a production-ready EditBrief - one
structured-output LLM call, using the idea's own concept/rationale/script plus the brand's
compliance rules. Never regenerates the idea itself; only elaborates it into scenes a video
editor (human or AI) can work from directly.

The JSON shape this produces (see edit_brief_to_json) is documented in README.md under
"Edit brief format" - it's the intended handoff format to external AI video tools, so changing
field names/shapes here is a breaking change for that contract, not just an internal refactor.
"""
import json
import re
from dataclasses import asdict
from typing import Optional

from core.models import BriefScene, CreativeIdea, EditBrief
from core.reasoning.engine import strip_code_fence
from core.reasoning.llm_client import LLMClient

EDIT_BRIEF_SYSTEM_PROMPT = (
    "You are a video editor's assistant. You are given a creative idea that has already been "
    "approved - do not change its concept, format, or core message - and that brand's "
    "non-negotiable compliance rules. Turn the idea into a scene-by-scene production brief:\n"
    "- aspect_ratio: default \"9:16\" unless the format clearly implies otherwise.\n"
    "- duration_seconds: the total runtime this brief's scenes will cover (an integer).\n"
    "- alternative_hooks: 2-3 different opening lines/hooks the editor or brand team could "
    "swap in instead of the scripted one, same tease-vs-thesis framing as the original.\n"
    "- scenes: an ordered, contiguous list of beats covering the entire duration with no gaps "
    "or overlaps - the first scene starts at 0:00, each next scene starts exactly where the "
    "previous one ended, and the last scene ends exactly at duration_seconds. Each scene has:\n"
    "  - timestamp_range: e.g. \"0:00-0:05\".\n"
    "  - beat_type: exactly one of \"hook\", \"body\", \"cta\".\n"
    "  - voiceover_or_onscreen_text: what's said or shown as text in this scene.\n"
    "  - visual_description: what the camera/footage shows.\n"
    "  - b_roll_suggestions: a list of specific supplementary shots, empty list if none.\n"
    "  - caption_text: the on-screen caption/subtitle for this scene, if any (empty string if none).\n"
    "  - footage_note: a specific note on what footage still needs to be shot for this scene "
    "(e.g. \"needs UGC selfie shot\"), or null if the brand's existing library likely covers it.\n"
    "- music_mood_note: a short note on the intended music/audio mood.\n\n"
    "The compliance rules apply to every scene's voiceover/on-screen text/captions exactly as "
    "much as they applied to the original idea - never soften or reinterpret them while "
    "elaborating detail. Respond with JSON only, no other text, matching this exact shape: "
    '{"aspect_ratio": "...", "duration_seconds": 0, "alternative_hooks": ["..."], '
    '"scenes": [{"timestamp_range": "...", "beat_type": "...", "voiceover_or_onscreen_text": "...", '
    '"visual_description": "...", "b_roll_suggestions": ["..."], "caption_text": "...", '
    '"footage_note": null}], "music_mood_note": "..."}'
)

EDIT_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "aspect_ratio": {"type": "string"},
        "duration_seconds": {"type": "integer"},
        "alternative_hooks": {"type": "array", "items": {"type": "string"}},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp_range": {"type": "string"},
                    "beat_type": {"type": "string", "enum": ["hook", "body", "cta"]},
                    "voiceover_or_onscreen_text": {"type": "string"},
                    "visual_description": {"type": "string"},
                    "b_roll_suggestions": {"type": "array", "items": {"type": "string"}},
                    "caption_text": {"type": "string"},
                    "footage_note": {"type": "string", "nullable": True},
                },
                "required": ["timestamp_range", "beat_type", "voiceover_or_onscreen_text", "visual_description"],
            },
        },
        "music_mood_note": {"type": "string"},
    },
    "required": ["aspect_ratio", "duration_seconds", "alternative_hooks", "scenes", "music_mood_note"],
}

# Small tolerance (seconds) for a scene boundary/final duration match - the model reasons in
# whole beats, not floating-point precision, so an off-by-a-hair timestamp shouldn't fail an
# otherwise-good brief.
_TIMESTAMP_TOLERANCE_SECONDS = 0.5


class EditBriefError(Exception):
    """The model's response couldn't be parsed into the expected shape, or its scenes don't
    actually validate (not contiguous, or don't sum to the stated duration)."""


def _parse_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unparseable timestamp: {value!r}")


def _parse_timestamp_range(value: str) -> tuple[float, float]:
    try:
        start_str, end_str = value.split("-")
        return _parse_timestamp(start_str), _parse_timestamp(end_str)
    except ValueError as e:
        raise ValueError(f"Unparseable timestamp range: {value!r}") from e


def validate_edit_brief(brief: EditBrief) -> None:
    """Raises EditBriefError if the scenes aren't contiguous from 0:00, don't collectively
    span exactly `duration_seconds`, or a scene's range doesn't move forward."""
    if not brief.scenes:
        raise EditBriefError("Edit brief has no scenes.")

    expected_start = 0.0
    for i, scene in enumerate(brief.scenes, start=1):
        try:
            start, end = _parse_timestamp_range(scene.timestamp_range)
        except ValueError as e:
            raise EditBriefError(f"Scene {i}: {e}") from e
        if end <= start:
            raise EditBriefError(f"Scene {i}'s timestamp range doesn't move forward: {scene.timestamp_range!r}")
        if abs(start - expected_start) > _TIMESTAMP_TOLERANCE_SECONDS:
            raise EditBriefError(
                f"Scene {i} starts at {start}s but scene {i - 1} ended at {expected_start}s - "
                "scenes must be contiguous, with no gap or overlap."
            )
        expected_start = end

    if abs(expected_start - brief.duration_seconds) > _TIMESTAMP_TOLERANCE_SECONDS:
        raise EditBriefError(
            f"Scenes end at {expected_start}s but the brief's duration_seconds is "
            f"{brief.duration_seconds} - they must match."
        )


def generate_edit_brief(
    idea: CreativeIdea, brand_name: str, compliance_rules: str, llm_client: LLMClient
) -> EditBrief:
    user_prompt = (
        f"Brand: {brand_name}\n"
        f"Approved idea concept: {idea.concept}\n"
        f"Rationale: {idea.rationale}\n"
        f"Format: {idea.recommended_format}\n"
        f"Script/brief so far: {idea.script or '(none provided - design a suitable scene structure for this concept)'}\n\n"
        f"Brand's non-negotiable compliance rules:\n{compliance_rules}"
    )
    raw = llm_client.generate(EDIT_BRIEF_SYSTEM_PROMPT, user_prompt, response_schema=EDIT_BRIEF_SCHEMA)

    try:
        parsed = json.loads(strip_code_fence(raw))
        scenes = [
            BriefScene(
                timestamp_range=s["timestamp_range"],
                beat_type=s["beat_type"],
                voiceover_or_onscreen_text=s["voiceover_or_onscreen_text"],
                visual_description=s["visual_description"],
                b_roll_suggestions=list(s.get("b_roll_suggestions") or []),
                caption_text=s.get("caption_text") or "",
                footage_note=s.get("footage_note"),
            )
            for s in parsed["scenes"]
        ]
        brief = EditBrief(
            concept=idea.concept,
            brand=brand_name,
            format=idea.recommended_format,
            aspect_ratio=parsed.get("aspect_ratio") or "9:16",
            duration_seconds=int(parsed["duration_seconds"]),
            alternative_hooks=list(parsed.get("alternative_hooks") or []),
            scenes=scenes,
            music_mood_note=parsed.get("music_mood_note") or "",
            compliance_notes=idea.review_reason if idea.needs_review else None,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise EditBriefError(f"Couldn't parse the model's edit-brief response: {e}") from e

    validate_edit_brief(brief)
    return brief


def edit_brief_to_json(brief: EditBrief) -> str:
    """Machine-readable form - the handoff format to external AI video tools (see README's
    "Edit brief format")."""
    return json.dumps(asdict(brief), indent=2)


def edit_brief_to_markdown(brief: EditBrief) -> str:
    """Human-readable form - a scene-by-scene table plus the surrounding notes."""
    lines = [
        f"# Edit brief: {brief.concept}",
        "",
        f"**Brand:** {brief.brand}  ",
        f"**Format:** {brief.format}  ",
        f"**Aspect ratio:** {brief.aspect_ratio}  ",
        f"**Duration:** {brief.duration_seconds}s",
        "",
    ]

    if brief.compliance_notes:
        lines += ["## ⚠️ Compliance note", "", brief.compliance_notes, ""]

    if brief.alternative_hooks:
        lines += ["## Alternative hooks", ""]
        lines += [f"- {hook}" for hook in brief.alternative_hooks]
        lines.append("")

    lines += [
        "## Scenes",
        "",
        "| Time | Beat | Voiceover / on-screen text | Visual | B-roll | Footage note |",
        "|---|---|---|---|---|---|",
    ]
    for scene in brief.scenes:
        b_roll = "; ".join(scene.b_roll_suggestions)
        lines.append(
            f"| {scene.timestamp_range} | {scene.beat_type} | {scene.voiceover_or_onscreen_text} "
            f"| {scene.visual_description} | {b_roll} | {scene.footage_note or ''} |"
        )
    lines.append("")

    lines += ["## Music / mood", "", brief.music_mood_note or "(none specified)"]

    captions = [(s.timestamp_range, s.caption_text) for s in brief.scenes if s.caption_text]
    if captions:
        lines += ["", "## Captions", ""]
        lines += [f"- **{ts}:** {caption}" for ts, caption in captions]

    return "\n".join(lines)


def slugify_for_filename(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug[:40].rstrip("-")) or "edit-brief"
