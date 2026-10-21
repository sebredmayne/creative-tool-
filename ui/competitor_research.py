"""Main-content-area panel: "what are the top brands in this category doing right now,"
backed by core/competitor_research.py. Unlike ui/ad_inspiration.py's sidebar panel (a quick
supplementary check for one already-known brand), this returns a much larger structure meant
to be read *before* chatting, so it lives in the main content area, collapsed by default once a
search completes so it doesn't crowd out the chat below it.

Deliberately standalone, same principle as ad_inspiration.py: results are shown for a marketer
to read and carry into the existing chat-based idea generator by hand, never fed automatically
into generation.
"""
import streamlit as st

from core.ad_library import AdExample, is_configured as meta_configured
from core.competitor_research import BrandAdFindings, CategoryResearchResult, research_category
from core.google_ads import GoogleAdExample, is_configured as google_configured
from core.reasoning.grounded_search import is_available as grounding_available
from core.reasoning.trends_synthesis import TrendsSummary


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_research(category: str, _llm_client) -> CategoryResearchResult:
    # Leading underscore on _llm_client excludes it from Streamlit's cache key (it isn't
    # hashable) - same convention as ad_inspiration.py's _cached_search.
    return research_category(category, _llm_client)


def _status_caption() -> str:
    parts = ["Meta: configured" if meta_configured() else "Meta: not set up - add META_AD_LIBRARY_ACCESS_TOKEN"]
    if not google_configured():
        parts.append("Google: not set up - add APIFY_API_TOKEN")
    elif not grounding_available():
        parts.append("Google: needs GEMINI_API_KEY (for finding a brand's advertiser page)")
    else:
        parts.append("Google: configured")
    return " · ".join(parts)


def _render_meta_ad(ad: AdExample) -> None:
    status = f"🟢 running {ad.days_running}d" if ad.is_active and ad.days_running is not None else "⏸️ stopped"
    st.markdown(f"**Meta** · {status}")
    st.caption(ad.body_text[:280])
    if ad.snapshot_url:
        st.markdown(f"[View in Meta Ad Library]({ad.snapshot_url})")


def _render_google_ad(ad: GoogleAdExample) -> None:
    status = "🟢 recent" if ad.is_active else "⏸️ older"
    st.markdown(f"**Google** ({ad.format.title()}) · {status}")
    st.caption(ad.headline_or_description[:280])
    if ad.preview_url:
        st.markdown(f"[View preview]({ad.preview_url})")


def _render_brand_findings(findings: BrandAdFindings) -> None:
    total_ads = len(findings.meta_ads) + len(findings.google_ads)
    with st.expander(f"{findings.brand_name} ({total_ads} ads found)"):
        if findings.meta_error:
            st.caption(f"Meta: {findings.meta_error}")
        for ad in findings.meta_ads:
            _render_meta_ad(ad)
            st.divider()

        if findings.google_skipped_reason:
            st.caption(f"Google: {findings.google_skipped_reason}")
        for ad in findings.google_ads:
            _render_google_ad(ad)
            st.divider()

        if not findings.meta_ads and not findings.google_ads and not findings.meta_error and not findings.google_skipped_reason:
            st.caption("No ads found for this brand.")


def _render_trends_summary(summary: TrendsSummary) -> None:
    st.markdown("##### What's trending")
    st.write(summary.overall_summary)
    if summary.patterns:
        st.markdown("**Patterns across brands**")
        for pattern in summary.patterns:
            st.markdown(f"- {pattern}")
    if summary.gaps_or_opportunities:
        st.markdown("**Possible gaps/opportunities**")
        for gap in summary.gaps_or_opportunities:
            st.markdown(f"- {gap}")
    if summary.per_brand_notes:
        with st.expander("Per-brand notes"):
            for note in summary.per_brand_notes:
                st.markdown(f"**{note.brand_name}**: {note.note}")


def render_competitor_research_panel(default_category: str = "", llm_client=None) -> None:
    """Call from app.py's main content area, before the example-prompts block. `default_category`
    seeds the input from the brand context's category field, same convention as
    render_ad_inspiration_panel's default_query. `llm_client` should be the same client the rest
    of the app uses (get_llm_client())."""
    with st.expander("🔎 Competitor ad research", expanded=False):
        st.caption(
            "Tell us a category or product, and we'll find the top brands advertising in it and "
            "show what's trending - read this first, then use the chat below to generate your own "
            "ideas informed by it."
        )
        st.caption(_status_caption())

        category = st.text_input("Category or product", value=default_category, key="competitor_research_category")
        if st.button("Research this category", key="competitor_research_search") and category.strip():
            with st.status(f'Researching "{category.strip()}"...', expanded=True) as status:
                st.write("Finding the top brands...")
                st.write("Pulling their Meta and Google ads (the slow part, up to ~30s)...")
                st.write("Summarizing what's trending...")
                result = _cached_research(category.strip(), llm_client)
                status.update(label="Research complete", state="complete", expanded=False)
            st.session_state["competitor_research_result"] = result

        # Stored in session_state, not just shown right after the button click - Streamlit
        # reruns this whole panel on every unrelated interaction elsewhere in the app (e.g.
        # sending a chat message), and results should survive that rerun instead of vanishing.
        result: CategoryResearchResult = st.session_state.get("competitor_research_result")
        if result is not None:
            if result.error:
                st.warning(result.error)
            else:
                if result.trends_summary:
                    _render_trends_summary(result.trends_summary)
                    st.divider()
                st.markdown("##### By brand")
                for findings in result.brands:
                    _render_brand_findings(findings)

        st.caption(
            "This summarizes competitors' own public ad claims, unverified - not medical or "
            "regulatory guidance. Don't carry a competitor's claim into your own ad without "
            "confirming it's true for your product and clearing it with your medical/regulatory "
            "reviewer."
        )
