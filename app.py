"""V0 Streamlit UI for the D2C Growth Copilot - brand picker + chat interface.

This file only handles widgets and rendering - one CopilotSession (core/session.py) owns all
state and business logic (brand selection, ingestion, chat history, generation), and ui/
holds the render functions and CSS theme. core/ never imports streamlit.
"""
# Streamlit Community Cloud's system sqlite3 is often older than what Chroma requires - swap
# in pysqlite3-binary's bundled modern SQLite before anything (chromadb, transitively) imports
# the stdlib sqlite3 module, so this has to run before any other import that could reach it.
# requirements.txt marks pysqlite3-binary Linux-only (it ships no macOS/Windows wheel), so the
# import fails harmlessly everywhere else, local dev included - this is a no-op there.
try:
    import pysqlite3
    import sys

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import logging
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="D2C Growth Copilot (V0)", layout="wide")

from dotenv import load_dotenv

load_dotenv()


def _load_secrets_into_env() -> None:
    """Streamlit Cloud provides secrets via st.secrets, not a .env file - copy the ones core/
    cares about into os.environ (without overwriting a real env var, e.g. one already set from
    local .env) so every core/ module keeps reading plain os.getenv() and stays
    Streamlit-free. A no-op locally if no .streamlit/secrets.toml exists at all.

    Deliberately excludes LLM_CACHE: that should only ever come from a deliberate local .env,
    never a secrets store, so the dev cache defaults off in production regardless of what's in
    st.secrets (see README's "Deploying" section).
    """
    try:
        secret_keys = list(st.secrets.keys())
    except FileNotFoundError:
        return  # no secrets.toml at all - nothing to load, e.g. plain local dev
    for key in ("GEMINI_API_KEY", "GEMINI_MODELS", "GROQ_API_KEY", "GROQ_MODEL", "DEMO_PASSWORD"):
        if key in secret_keys:
            os.environ.setdefault(key, str(st.secrets[key]))


_load_secrets_into_env()

from core.custom_brands import delete_custom_brand, get_brand, list_all_brands
from core.knowledge.loader import load_explore_knowledge
from core.reasoning.llm_client import get_llm_client
from core.retrieval.vector_store import VectorStore
from core.session import OTHER_BRAND_KEY, CopilotSession, UnsupportedFileTypeError
from ui.brand_form import render_creation_flow
from ui.components import render_assistant_message
from ui.theme import THEME_CSS

LOG_DIR = Path(__file__).parent / "logs"

logger = logging.getLogger("d2c_growth_copilot")
if not logger.handlers:
    # Streamlit re-executes this whole script on every rerun - guard against
    # attaching duplicate handlers (and therefore duplicate log lines) each time.
    LOG_DIR.mkdir(exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_DIR / "app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)

st.markdown(THEME_CSS, unsafe_allow_html=True)


def _check_demo_password() -> bool:
    """Gates the whole app behind a shared password if DEMO_PASSWORD is set (secrets or env) -
    protects a public demo link's free-tier API key from being hammered by strangers. A no-op
    (always allowed) if DEMO_PASSWORD isn't set at all, e.g. local dev or a private deployment.
    """
    required_password = os.getenv("DEMO_PASSWORD")
    if not required_password or st.session_state.get("demo_password_ok"):
        return True

    st.title("D2C Growth Copilot")
    st.caption("This demo is password-protected - ask whoever shared this link for the password.")
    entered = st.text_input("Password", type="password", key="demo_password_input")
    if st.button("Enter"):
        if entered == required_password:
            st.session_state.demo_password_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not _check_demo_password():
    st.stop()


@st.cache_resource
def get_vector_store() -> VectorStore:
    store = VectorStore()
    # Connect collections are per-session (connect_<uuid>) so uploads never leak between
    # sessions - but a session that ends without cleanup (crash, restart, closed tab) leaves
    # its collection behind on disk forever otherwise. This runs once per process, before any
    # session creates its own collection, so it only ever clears genuinely stale ones.
    store.delete_collections_with_prefix("connect_")
    load_explore_knowledge(store)
    return store


@st.cache_resource
def get_llm():
    return get_llm_client()


if "session" not in st.session_state:
    st.session_state.session = CopilotSession(get_vector_store(), get_llm())
session: CopilotSession = st.session_state.session
vector_store = get_vector_store()
llm_client = get_llm()

if "new_brand_slug" in st.session_state:
    # A brand was just saved (ui/brand_form.py already rebuilt Explore knowledge for it) -
    # jump straight into it rather than dropping the marketer back on the picker.
    session.load_brand(st.session_state.pop("new_brand_slug"))
    st.rerun()

st.title("D2C Growth Copilot")

if st.session_state.get("brand_creation_step"):
    render_creation_flow(vector_store, llm_client)
elif session.selected_brand is None:
    st.markdown("#### How to use this")
    st.markdown(
        "1. Pick a brand card below (or skip) - it loads sample data and brand context "
        "for you automatically, no typing needed.\n"
        "2. Type what you want in the chat box at the bottom - a script, ad concepts, "
        "campaign ideas, anything.\n"
        "3. Get back structured ideas, each with its own reasoning and the source it "
        "actually drew from."
    )

    with st.expander("🔧 What happens behind the scenes"):
        st.markdown(
            "When you send a message, the app searches a knowledge base - curated brand "
            "info plus whatever data you loaded - for the most relevant pieces, then sends "
            "those along with your question to Google's Gemini AI to write a real answer. "
            "Nothing gets called until you actually hit send."
        )

    st.write("")
    st.markdown("### Which brand are we working on?")
    st.caption(
        "Picking one opens a chat with its catalogue, reviews, and past campaigns loaded. "
        "Every idea says where it came from."
    )

    def render_pickable_brand_card(key: str, profile) -> None:
        with st.container(border=True):
            st.caption(profile.label.upper())
            badge_text = "Custom brand" if profile.is_custom else "Brand data"
            st.markdown(f'<span class="idea-badge badge-brand-data">{badge_text}</span>', unsafe_allow_html=True)
            st.markdown(f'<p class="card-title">{profile.brand_name.split(" / ")[0]}</p>', unsafe_allow_html=True)
            st.write(profile.description)
            if st.button("Start chat →", key=f"pick_{key}", type="tertiary"):
                session.load_brand(key)
                st.rerun()
            if profile.is_custom and st.button("Delete", key=f"delete_{key}", use_container_width=True):
                delete_custom_brand(key)
                load_explore_knowledge(vector_store)
                st.rerun()

    def render_other_brand_card() -> None:
        with st.container(border=True):
            st.caption("OTHER")
            st.markdown('<span class="idea-badge badge-inference">General only</span>', unsafe_allow_html=True)
            st.markdown('<p class="card-title">Another brand</p>', unsafe_allow_html=True)
            st.write("No brand data. Ideas use general D2C knowledge.")
            if st.button("Start chat →", key="pick_other", type="tertiary"):
                session.set_other_brand()
                st.rerun()

    def render_add_brand_card() -> None:
        with st.container(border=True):
            st.caption("NEW")
            st.markdown('<span class="idea-badge badge-inference">Bring your own</span>', unsafe_allow_html=True)
            st.markdown('<p class="card-title">+ Add your brand</p>', unsafe_allow_html=True)
            st.write("Upload a brand guide - we'll extract voice, positioning, personas, and compliance rules.")
            if st.button("Get started →", key="pick_add_brand", type="tertiary"):
                st.session_state.brand_creation_step = "form"
                st.rerun()

    # Presets first, then any saved custom brands, then the two fixed cards - laid out in rows
    # of 4 so this scales past exactly-4 without a hardcoded column count.
    pickable = list(list_all_brands().items()) + [("__add_brand__", None), ("__other_brand__", None)]
    for row_start in range(0, len(pickable), 4):
        row = pickable[row_start : row_start + 4]
        cols = st.columns(4)
        for col, (key, profile) in zip(cols, row):
            with col:
                if key == "__add_brand__":
                    render_add_brand_card()
                elif key == "__other_brand__":
                    render_other_brand_card()
                else:
                    render_pickable_brand_card(key, profile)

else:
    with st.sidebar:
        col_brand, col_change = st.columns([3, 2])
        with col_brand:
            if session.selected_brand != OTHER_BRAND_KEY:
                profile = get_brand(session.selected_brand)
                brand_label = profile.brand_name.split(" / ")[0]
                badge_text = "Custom brand" if profile.is_custom else "Brand data"
                st.markdown(
                    f'**{brand_label}** <span class="idea-badge badge-brand-data">{badge_text}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '**Other brand** <span class="idea-badge badge-inference">General only</span>',
                    unsafe_allow_html=True,
                )
        with col_change:
            if st.button("← Change", key="change_brand", use_container_width=True):
                session.selected_brand = None
                st.rerun()

        if st.button("+ New chat", use_container_width=True):
            session.new_chat()
            st.rerun()

        if session.past_chats:
            with st.expander(f"Past chats ({len(session.past_chats)})"):
                for i, past in enumerate(session.past_chats):
                    if st.button(past["title"] or f"Chat {i + 1}", key=f"pastchat_{i}", use_container_width=True):
                        session.restore_chat(i)
                        st.rerun()

        st.divider()
        st.caption("Data in use")
        if session.loaded_files:
            for filename in list(session.loaded_files):
                col_name, col_remove = st.columns([5, 1])
                with col_name:
                    tag = " `sample`" if filename.startswith("[sample]") else ""
                    st.markdown(f"{filename}{tag}")
                with col_remove:
                    if st.button("×", key=f"remove_{filename}"):
                        session.remove_file(filename)
                        st.rerun()
        else:
            st.caption("No data loaded - using general knowledge only.")

        with st.expander("+ Add your own · CSV, XLSX, PDF"):
            uploaded = st.file_uploader(
                "Upload", type=["csv", "xlsx", "pdf"], accept_multiple_files=True, label_visibility="collapsed"
            )
            if uploaded:
                for file in uploaded:
                    if file.name not in session.loaded_files:
                        try:
                            session.ingest(file, file.name)
                        except UnsupportedFileTypeError:
                            st.warning(f"Unsupported file type: {file.name}")
                st.rerun()

        if session.loaded_files and session.selected_brand != OTHER_BRAND_KEY:
            if st.button("Reset to sample data", key="reset_data"):
                session.reset_to_sample()
                st.rerun()

        with st.expander("Brand context"):
            preset_context = session.brand_context
            brand = st.text_input("Brand", value=preset_context.get("brand", ""))
            product = st.text_input("Product", value=preset_context.get("product", ""))
            customer = st.text_input("Target customer", value=preset_context.get("customer", ""))
            category = st.text_input("Category", value=preset_context.get("category", ""))
            objective = st.text_input("Business objective", value=preset_context.get("objective", ""))
            st.caption("This copilot avoids medical/efficacy claims and flags anything that needs regulatory review.")

        show_comparison = st.checkbox("Also show generic comparison (extra API call)")

        if session.saved_ideas:
            st.divider()
            with st.expander(f"💾 Saved ideas ({len(session.saved_ideas)})"):
                for saved in session.saved_ideas:
                    st.markdown(f"**{saved.concept}**")
                    st.caption(saved.rationale[:140])
                if st.button("Clear saved ideas", use_container_width=True):
                    session.clear_saved_ideas()
                    st.rerun()

    _GENERIC_EXAMPLE_PROMPTS = [
        {"category": "Reel scripts", "prompt": "Give me 5 Instagram Reel concepts for a new D2C brand"},
        {"category": "Ad concepts", "prompt": "Give me ad angles based on common category objections"},
        {"category": "Campaign", "prompt": "Give me a launch campaign idea for a new D2C brand"},
    ]

    if session.selected_brand != OTHER_BRAND_KEY:
        profile = get_brand(session.selected_brand)
        example_prompts = profile.example_prompts or _GENERIC_EXAMPLE_PROMPTS
        brand_display_name = profile.brand_name.split(" / ")[0]
    else:
        example_prompts = _GENERIC_EXAMPLE_PROMPTS
        brand_display_name = "your brand"

    use_example_prompt = None
    if not session.messages:
        st.markdown(f"#### What do you want to make for {brand_display_name}?")
        st.caption("Not sure what to ask? Start with one of these.")
        ex_cols = st.columns(3)
        for col, ex in zip(ex_cols, example_prompts):
            with col:
                with st.container(border=True):
                    st.caption(ex["category"].upper())
                    st.write(ex["prompt"])
                    if st.button("Use this →", key=f"ex_{ex['category']}", type="tertiary"):
                        use_example_prompt = ex["prompt"]

    pending_query = None
    for msg_idx, message in enumerate(session.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                result = render_assistant_message(message, key_prefix=f"msg{msg_idx}", session=session)
                if result:
                    pending_query = result
            else:
                st.markdown(message["content"])

    st.caption(
        "Ideas are drafts grounded in fictional sample brands. Any customer-facing copy still "
        "needs human and medical/regulatory review before use."
    )
    typed_query = st.chat_input("Ask for a script, ad concepts, campaign ideas...")
    query = pending_query or use_example_prompt or typed_query

    if query:
        brand_context = {
            "brand": brand,
            "product": product,
            "customer": customer,
            "category": category,
            "objective": objective,
        }
        with st.spinner("Thinking..."):
            session.ask(query, brand_context, compare_generic=show_comparison)
        st.rerun()
