"""Render functions for chat replies - idea cards, follow-up chips, and the assistant message
they're assembled into. Pure rendering plus widget event handling; any state change a button
triggers goes through the CopilotSession passed in, never a bare st.session_state mutation."""
import html

import streamlit as st

from core.reasoning.edit_brief import edit_brief_to_json, edit_brief_to_markdown, slugify_for_filename
from core.session import CopilotSession

# Free-text idea.recommended_format values that mean "this is a video, not a static/carousel
# ad" - "Export edit brief" only makes sense for those. Keyword match, not exhaustive: matches
# the same free-text-format convention the model already uses (see OUTPUT_INSTRUCTIONS).
_VIDEO_FORMAT_KEYWORDS = ("reel", "video", "ugc", "tiktok", "short")


def _is_video_format(format_text: str) -> bool:
    lowered = format_text.lower()
    return any(keyword in lowered for keyword in _VIDEO_FORMAT_KEYWORDS)


def _render_edit_brief(brief, key_prefix: str) -> None:
    st.markdown("###### Edit brief")
    st.caption(f"{brief.aspect_ratio} · {brief.duration_seconds}s · {brief.format}")

    if brief.compliance_notes:
        st.markdown(f'<div class="compliance-warning">⚠️ {html.escape(brief.compliance_notes)}</div>', unsafe_allow_html=True)

    if brief.alternative_hooks:
        with st.expander("Alternative hooks"):
            for hook in brief.alternative_hooks:
                st.markdown(f"- {hook}")

    st.dataframe(
        [
            {
                "Time": scene.timestamp_range,
                "Beat": scene.beat_type,
                "VO / on-screen text": scene.voiceover_or_onscreen_text,
                "Visual": scene.visual_description,
                "B-roll": "; ".join(scene.b_roll_suggestions),
                "Footage note": scene.footage_note or "",
            }
            for scene in brief.scenes
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Music/mood: {brief.music_mood_note}" if brief.music_mood_note else "")

    filename_stub = slugify_for_filename(brief.concept)
    col_json, col_md = st.columns(2)
    with col_json:
        st.download_button(
            "Download JSON",
            data=edit_brief_to_json(brief),
            file_name=f"{filename_stub}_edit_brief.json",
            mime="application/json",
            key=f"download_json_{key_prefix}",
            use_container_width=True,
        )
    with col_md:
        st.download_button(
            "Download Markdown",
            data=edit_brief_to_markdown(brief),
            file_name=f"{filename_stub}_edit_brief.md",
            mime="text/markdown",
            key=f"download_md_{key_prefix}",
            use_container_width=True,
        )


def _describe_idea_fully(idea) -> str:
    """Renders one idea's full content (not just its concept name) for embedding into a
    follow-up query - so "make it funnier" or "refine this" identifies exactly which idea it
    means and what's actually in it, rather than a bare title the model has to guess back."""
    parts = [f'Concept: "{idea.concept}"', f"Format: {idea.recommended_format}", f"Rationale: {idea.rationale}"]
    if idea.script:
        parts.append(f"Script: {idea.script}")
    return "\n".join(parts)


def render_idea_card(idea, key_prefix: str, session: CopilotSession):
    """Renders one idea card. Returns a follow-up query string if Refine was clicked, else None."""
    badge_class = "badge-brand-data" if idea.grounding == "brand_data" else "badge-inference"
    badge_text = "Brand data" if idea.grounding == "brand_data" else "Inference"

    follow_up = None
    with st.container(border=True):
        col_title, col_badge = st.columns([4, 1])
        with col_title:
            st.markdown(f'<p class="idea-title">{html.escape(idea.concept)}</p>', unsafe_allow_html=True)
        with col_badge:
            st.markdown(f'<span class="idea-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

        if idea.needs_review:
            reason = idea.review_reason or "Needs medical/regulatory review before use."
            st.markdown(f'<div class="compliance-warning">⚠️ {html.escape(reason)}</div>', unsafe_allow_html=True)

        with st.expander("Why this"):
            st.markdown(f"**Rationale:** {idea.rationale}")
            st.markdown(f"**Recommended format:** {idea.recommended_format}")
            st.markdown(f"**Source/context used:** {idea.source_context}")

        # st.code renders with its own built-in copy-to-clipboard icon on hover - no custom
        # button/JS needed (Streamlit strips inline onclick handlers from st.markdown HTML
        # anyway, which is why the old copy button never actually worked).
        st.code(idea.script or idea.rationale, language=None, wrap_lines=True)

        if _is_video_format(idea.recommended_format):
            col_refine, col_save, col_brief = st.columns(3)
        else:
            col_refine, col_save = st.columns(2)
            col_brief = None

        with col_refine:
            if st.button("Refine", key=f"refine_{key_prefix}", use_container_width=True):
                follow_up = f"Give me a different take on this idea:\n{_describe_idea_fully(idea)}"
        with col_save:
            if st.button("Save", key=f"save_{key_prefix}", use_container_width=True):
                session.save_idea(idea)
                st.toast(f"Saved: {idea.concept}")
        if col_brief is not None:
            with col_brief:
                if st.button("Export edit brief", key=f"brief_btn_{key_prefix}", use_container_width=True):
                    with st.spinner("Generating edit brief..."):
                        result = session.generate_edit_brief(idea)
                    if result.error:
                        st.error(result.error)
                    else:
                        st.session_state[f"edit_brief_{key_prefix}"] = result.brief

        if col_brief is not None and st.session_state.get(f"edit_brief_{key_prefix}"):
            _render_edit_brief(st.session_state[f"edit_brief_{key_prefix}"], key_prefix)

    return follow_up


def render_idea_cards(ideas, key_prefix: str, session: CopilotSession):
    """Renders a list of idea cards. Returns a follow-up query string if any Refine was clicked."""
    follow_up = None
    for i, idea in enumerate(ideas):
        result = render_idea_card(idea, key_prefix=f"{key_prefix}_{i}", session=session)
        if result:
            follow_up = result
    return follow_up


def render_followup_chips(ideas, key_prefix: str):
    """Suggested follow-ups shown under a reply. Returns a follow-up query if one was clicked."""
    if not ideas:
        return None
    chips = []
    if len(ideas) >= 2:
        chips.append(("Make idea 2 funnier", f"Make this idea funnier:\n{_describe_idea_fully(ideas[1])}"))
    chips.append(("Write the full script for idea 1", f'Write a complete, detailed script for: "{ideas[0].concept}"'))
    chips.append(("Hindi versions", "Give me Hindi versions of these same ideas"))

    cols = st.columns(len(chips))
    follow_up = None
    for col, (label, prompt) in zip(cols, chips):
        with col:
            if st.button(label, key=f"chip_{key_prefix}_{label}", use_container_width=True):
                follow_up = prompt
    return follow_up


def render_assistant_message(message: dict, key_prefix: str, session: CopilotSession):
    if message.get("error"):
        st.error(message["error"])
        return None
    if message.get("generic_ideas"):
        col_generic, col_data = st.columns(2)
        with col_generic:
            st.markdown("#### Generic (no data)")
            follow_up_a = render_idea_cards(message["generic_ideas"], key_prefix=f"{key_prefix}_generic", session=session)
        with col_data:
            st.markdown("#### With your data")
            follow_up_b = render_idea_cards(message["ideas"], key_prefix=f"{key_prefix}_data", session=session)
        return follow_up_a or follow_up_b
    follow_up = render_idea_cards(message["ideas"], key_prefix=key_prefix, session=session)
    return follow_up or render_followup_chips(message["ideas"], key_prefix=key_prefix)
