"""V0 Streamlit UI for the D2C Growth Copilot - brand picker + chat interface.

This file only handles widgets and rendering - one CopilotSession (core/session.py) owns all
state and business logic (brand selection, ingestion, chat history, generation), and ui/
holds the render functions and CSS theme. core/ never imports streamlit.
"""
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.brand_presets import BRAND_PRESETS
from core.knowledge.loader import load_explore_knowledge
from core.reasoning.llm_client import get_llm_client
from core.retrieval.vector_store import VectorStore
from core.session import OTHER_BRAND_KEY, CopilotSession, UnsupportedFileTypeError
from ui.components import render_assistant_message
from ui.theme import THEME_CSS

st.set_page_config(page_title="D2C Growth Copilot (V0)", layout="wide")

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

st.title("D2C Growth Copilot")

if session.selected_brand is None:
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

    cols = st.columns(4)
    for col, (key, preset) in zip(cols[:3], BRAND_PRESETS.items()):
        with col:
            with st.container(border=True):
                st.caption(preset["label"].upper())
                st.markdown('<span class="idea-badge badge-brand-data">Brand data</span>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="card-title">{preset["brand_name"].split(" / ")[0]}</p>', unsafe_allow_html=True
                )
                st.write(preset["description"])
                if st.button("Start chat →", key=f"pick_{key}", type="tertiary"):
                    session.load_brand(key)
                    st.rerun()

    with cols[3]:
        with st.container(border=True):
            st.caption("OTHER")
            st.markdown('<span class="idea-badge badge-inference">General only</span>', unsafe_allow_html=True)
            st.markdown('<p class="card-title">Another brand</p>', unsafe_allow_html=True)
            st.write("No brand data. Ideas use general D2C knowledge.")
            if st.button("Start chat →", key="pick_other", type="tertiary"):
                session.set_other_brand()
                st.rerun()

else:
    with st.sidebar:
        col_brand, col_change = st.columns([3, 2])
        with col_brand:
            if session.selected_brand != OTHER_BRAND_KEY:
                preset = BRAND_PRESETS[session.selected_brand]
                brand_label = preset["brand_name"].split(" / ")[0]
                st.markdown(
                    f'**{brand_label}** <span class="idea-badge badge-brand-data">Brand data</span>',
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

    if session.selected_brand != OTHER_BRAND_KEY:
        preset = BRAND_PRESETS[session.selected_brand]
        example_prompts = preset["example_prompts"]
        brand_display_name = preset["brand_name"].split(" / ")[0]
    else:
        example_prompts = [
            {"category": "Reel scripts", "prompt": "Give me 5 Instagram Reel concepts for a new D2C skincare brand"},
            {"category": "Ad concepts", "prompt": "Give me ad angles based on common category objections"},
            {"category": "Campaign", "prompt": "Give me a launch campaign idea for a new D2C brand"},
        ]
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
