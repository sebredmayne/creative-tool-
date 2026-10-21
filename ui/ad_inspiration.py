"""Sidebar panel for "what's running in this niche right now" ad inspiration, backed by
core/ad_library.py (Meta Ad Library API - no AI API involved). Deliberately standalone: results
are shown for a marketer to read and borrow ideas from by hand, never fed automatically into
generation, so there's no session/state coupling here beyond a plain cached search call."""
import streamlit as st

from core.ad_library import AdExample, is_configured, search_ads


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_search(query: str):
    result = search_ads(query)
    return result.ads, result.error


def _render_ad(ad: AdExample) -> None:
    status = f"🟢 running {ad.days_running}d" if ad.is_active and ad.days_running is not None else "⏸️ stopped"
    st.markdown(f"**{ad.page_name}** · {status}")
    st.caption(ad.body_text[:280])
    st.markdown(f"[View in Meta Ad Library]({ad.snapshot_url})")
    st.divider()


def render_ad_inspiration_panel(default_query: str = "") -> None:
    """Call from the sidebar once a brand is selected. `default_query` seeds the search box -
    pass the brand's category (e.g. "Kids nutrition") so a marketer isn't starting from blank."""
    with st.expander("📣 Ad inspiration"):
        if not is_configured():
            st.caption(
                "Not set up - add META_AD_LIBRARY_ACCESS_TOKEN to search live ads from Meta's "
                "Ad Library for inspiration."
            )
            return

        st.caption(
            "Ads currently running in this niche, via Meta's public Ad Library. "
            "\"Days running\" is a proxy for performance (advertisers cut losing ads fast) - "
            "not a real metric."
        )
        query = st.text_input("Search ads", value=default_query, key="ad_inspiration_query")
        if st.button("Search ads", key="ad_inspiration_search") and query.strip():
            st.session_state["ad_inspiration_results"] = _cached_search(query.strip())

        # Stored in session_state, not just shown right after the button click above - Streamlit
        # reruns this whole panel on every unrelated interaction elsewhere in the app (e.g.
        # sending a chat message), and results should survive that rerun instead of vanishing.
        results = st.session_state.get("ad_inspiration_results")
        if results is not None:
            ads, error = results
            if error:
                st.warning(error)
            elif not ads:
                st.caption("No active ads found for that search.")
            else:
                for ad in ads:
                    _render_ad(ad)
