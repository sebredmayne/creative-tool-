"""Shared data structures used across the ingestion, retrieval, and reasoning layers."""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ParsedDocument:
    """Output of a Parser: text for semantic search, and (if tabular) DataFrames for structured analysis."""

    filename: str
    file_type: str
    text_chunks: list[str]
    dataframes: Optional[dict[str, pd.DataFrame]] = None


@dataclass
class RetrievedChunk:
    """A single chunk returned by a vector store query."""

    text: str
    source: str
    distance: float


@dataclass
class QueryContext:
    """Everything needed to answer one user request, for either mode."""

    mode: str  # "explore" or "connect"
    query: str
    brand: str = ""
    product: str = ""
    customer: str = ""
    category: str = ""
    objective: str = ""


@dataclass
class CreativeIdea:
    """A single structured piece of V0 output."""

    concept: str
    rationale: str
    recommended_format: str
    source_context: str
    script: Optional[str] = None
    # Self-reported by the model, not inferred after the fact from prose:
    grounding: str = "inference"  # "brand_data" or "inference"
    needs_review: bool = False
    review_reason: Optional[str] = None


@dataclass
class BrandProfile:
    """One pickable brand - a curated preset (core/brand_presets.py) or a marketer-created
    custom brand (core/custom_brands.py) - same shape either way, so session loading,
    retrieval scoping, and compliance injection have exactly one code path regardless of
    which kind a given brand is.

    `voice_and_tone`/`positioning`/`personas`/`compliance_rules` are populated for custom
    brands (extracted from an uploaded guide, then user-edited - see
    core.reasoning.brand_extraction) and left empty for presets, whose equivalent content
    already lives in core/knowledge/explore/*.md and is retrieved the same way regardless.
    `compliance_rules` is the one field that matters for both: core.knowledge.loader.
    load_compliance_rules() reads it directly for custom brands, non-negotiable and
    deterministic either way.
    """

    slug: str
    label: str
    brand_name: str
    description: str
    context: dict  # brand/product/customer/category/objective - QueryContext's free-text fields
    voice_and_tone: str = ""
    positioning: str = ""
    personas: str = ""
    compliance_rules: str = ""
    sample_files: list[str] = field(default_factory=list)
    example_prompts: list[dict] = field(default_factory=list)
    guide_filename: str = ""
    is_custom: bool = False


@dataclass
class BriefScene:
    """One beat of a video edit brief - a production-ready instruction, not a paraphrase of
    the idea's script. `timestamp_range` is a "M:SS-M:SS" (or "SS-SS") string; scenes across
    one EditBrief must be contiguous and collectively span its duration_seconds (see
    core.reasoning.edit_brief.validate_edit_brief)."""

    timestamp_range: str
    beat_type: str  # "hook" | "body" | "cta"
    voiceover_or_onscreen_text: str
    visual_description: str
    b_roll_suggestions: list[str] = field(default_factory=list)
    caption_text: str = ""
    footage_note: Optional[str] = None  # e.g. "needs UGC selfie shot"


@dataclass
class EditBrief:
    """A production-ready brief a video editor (human or AI) can work from directly - one
    structured-output LLM call elaborating an already-generated CreativeIdea into scenes, never
    regenerating the idea itself (core.reasoning.edit_brief.generate_edit_brief).
    `compliance_notes` is carried through from the source idea's own review_reason, not
    re-derived here - the compliance *assessment* already happened during idea generation."""

    concept: str
    brand: str
    format: str
    duration_seconds: int
    scenes: list[BriefScene] = field(default_factory=list)
    alternative_hooks: list[str] = field(default_factory=list)
    aspect_ratio: str = "9:16"
    music_mood_note: str = ""
    compliance_notes: Optional[str] = None
