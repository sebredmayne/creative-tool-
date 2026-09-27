"""V0 Streamlit UI for the D2C Growth Copilot - brand picker + chat interface.

This file only handles UI state and wiring - all ingestion, retrieval, and
reasoning logic lives in core/.
"""
import html
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.brand_presets import BRAND_PRESETS
from core.ingestion import get_parser
from core.knowledge.loader import load_explore_knowledge
from core.models import QueryContext
from core.reasoning.engine import generate_ideas
from core.reasoning.llm_client import QuotaExceededError, get_llm_client
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import CONNECT_COLLECTION, VectorStore

st.set_page_config(page_title="D2C Growth Copilot (V0)", layout="wide")

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, p, span, label, li, div {
    font-family: 'Inter', sans-serif;
    color: #1A1512;
}

h1, h2, h3, h4, .card-title, .idea-title {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    color: #1A1512 !important;
}

.stApp {
    background-color: #FAF7F2;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E7DFD0 !important;
    border-radius: 14px !important;
    box-shadow: none;
}

/* Default (secondary) buttons: outlined pill - Refine/Save/Copy/Change/Reset etc. */
[data-testid="stBaseButton-secondary"] {
    background-color: #FFFFFF;
    color: #C1502E;
    border: 1px solid #C1502E;
    border-radius: 999px;
    font-weight: 600;
}
[data-testid="stBaseButton-secondary"]:hover {
    background-color: #FBF0EA;
    color: #A5401F;
    border-color: #A5401F;
}
[data-testid="stBaseButton-secondary"] p {
    color: inherit;
}

/* Tertiary buttons: flat text link, no border/fill - "Start chat ->", "Use this" */
[data-testid="stBaseButton-tertiary"] {
    background-color: transparent;
    color: #C1502E;
    border: none;
    font-weight: 700;
    padding-left: 0;
    justify-content: flex-start;
}
[data-testid="stBaseButton-tertiary"]:hover {
    color: #A5401F;
    text-decoration: underline;
}
[data-testid="stBaseButton-tertiary"] p {
    color: inherit;
}

.card-title {
    font-size: 1.3rem;
    margin-bottom: 2px;
}

.idea-title {
    font-size: 1.15rem;
    margin: 0;
    font-weight: 700 !important;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
}

.script-block {
    white-space: pre-wrap;
    word-wrap: break-word;
    background-color: #FBF8F2;
    border: 1px solid #E7DFD0;
    border-radius: 10px;
    padding: 14px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    line-height: 1.5;
}

.idea-badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    white-space: nowrap;
}
.badge-brand-data {
    background-color: #E3F0E3;
    color: #2F7D4F;
}
.badge-inference {
    background-color: #FBF0DC;
    color: #96700F;
}

.compliance-warning {
    background-color: #FBF0DC;
    border: 1px solid #E9D8A6;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #6B5410;
}

.copy-btn {
    width: 100%;
    background-color: #FFFFFF;
    color: #C1502E;
    border: 1px solid #C1502E;
    border-radius: 999px;
    font-weight: 600;
    padding: 0.4rem 0;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
}
.copy-btn:hover {
    background-color: #FBF0EA;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_vector_store() -> VectorStore:
    store = VectorStore()
    load_explore_knowledge(store)
    return store


@st.cache_resource
def get_llm():
    return get_llm_client()


def ingest_file(file, filename: str) -> None:
    """Parse a file and add it to the Connect vector store + structured store.

    Shared by real uploads and brand-preset sample data.
    """
    parser = get_parser(filename)
    if parser is None:
        st.warning(f"Unsupported file type: {filename}")
        return

    parsed = parser.parse(file, filename)
    ids = [f"{filename}-{i}" for i in range(len(parsed.text_chunks))]
    metadatas = [{"source": filename} for _ in parsed.text_chunks]
    vector_store.add_documents(CONNECT_COLLECTION, ids=ids, texts=parsed.text_chunks, metadatas=metadatas)

    if parsed.dataframes:
        for table_name, df in parsed.dataframes.items():
            st.session_state.structured_store.add(table_name, df)

    st.session_state.loaded_files.append(filename)


def load_brand_preset(key: str) -> None:
    preset = BRAND_PRESETS[key]
    for filename in preset["sample_files"]:
        sample_name = f"[sample] {filename}"
        if sample_name not in st.session_state.loaded_files:
            with open(SAMPLE_DATA_DIR / filename, "rb") as f:
                ingest_file(f, sample_name)
    st.session_state.selected_brand = key
    st.session_state.brand_context = preset["context"]


def describe_generation_error(exc: Exception) -> str:
    """Returns an error message to store on the message dict, not render directly - a message
    rendered here would vanish on the st.rerun() right after generation, since only the
    message dict persists across reruns."""
    if isinstance(exc, QuotaExceededError):
        return (
            "Gemini's free-tier quota for this model is used up for now (a hard daily limit, "
            "not a blip) - either wait for it to reset, or set a different GEMINI_MODEL in .env."
        )
    return "The model is temporarily unavailable (this is on Google's side, not this app). Please try again in a moment."


def render_idea_card(idea, key_prefix: str):
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

        if idea.script:
            st.markdown(f'<div class="script-block">{html.escape(idea.script)}</div>', unsafe_allow_html=True)

        col_refine, col_save, col_copy = st.columns(3)
        with col_refine:
            if st.button("Refine", key=f"refine_{key_prefix}", use_container_width=True):
                follow_up = f'Give me a different take on this idea: "{idea.concept}" - {idea.rationale}'
        with col_save:
            if st.button("Save", key=f"save_{key_prefix}", use_container_width=True):
                if idea not in st.session_state.saved_ideas:
                    st.session_state.saved_ideas.append(idea)
                st.toast(f"Saved: {idea.concept}")
        with col_copy:
            copy_text = idea.script or idea.rationale
            st.markdown(
                f'<button class="copy-btn" onclick="navigator.clipboard.writeText({json.dumps(copy_text)})">'
                "Copy</button>",
                unsafe_allow_html=True,
            )

    return follow_up


def render_idea_cards(ideas, key_prefix: str):
    """Renders a list of idea cards. Returns a follow-up query string if any Refine was clicked."""
    follow_up = None
    for i, idea in enumerate(ideas):
        result = render_idea_card(idea, key_prefix=f"{key_prefix}_{i}")
        if result:
            follow_up = result
    return follow_up


if "structured_store" not in st.session_state:
    st.session_state.structured_store = StructuredStore()
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_brand" not in st.session_state:
    st.session_state.selected_brand = None
if "brand_context" not in st.session_state:
    st.session_state.brand_context = {}
if "saved_ideas" not in st.session_state:
    st.session_state.saved_ideas = []
if "past_chats" not in st.session_state:
    st.session_state.past_chats = []

vector_store = get_vector_store()
llm_client = get_llm()

st.title("D2C Growth Copilot")

if st.session_state.selected_brand is None:
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
                    load_brand_preset(key)
                    st.rerun()

    with cols[3]:
        with st.container(border=True):
            st.caption("OTHER")
            st.markdown('<span class="idea-badge badge-inference">General only</span>', unsafe_allow_html=True)
            st.markdown('<p class="card-title">Another brand</p>', unsafe_allow_html=True)
            st.write("No brand data. Ideas use general D2C knowledge.")
            if st.button("Start chat →", key="pick_other", type="tertiary"):
                st.session_state.selected_brand = "none"
                st.rerun()

else:
    with st.sidebar:
        col_brand, col_change = st.columns([3, 2])
        with col_brand:
            if st.session_state.selected_brand != "none":
                preset = BRAND_PRESETS[st.session_state.selected_brand]
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
                st.session_state.selected_brand = None
                st.rerun()

        if st.button("+ New chat", use_container_width=True):
            if st.session_state.messages:
                first_user_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Chat")
                st.session_state.past_chats.append(
                    {
                        "title": first_user_msg[:40],
                        "brand": st.session_state.selected_brand,
                        "messages": st.session_state.messages,
                    }
                )
            st.session_state.messages = []
            st.rerun()

        if st.session_state.past_chats:
            with st.expander(f"Past chats ({len(st.session_state.past_chats)})"):
                for i, past in enumerate(st.session_state.past_chats):
                    if st.button(past["title"] or f"Chat {i + 1}", key=f"pastchat_{i}", use_container_width=True):
                        st.session_state.messages = past["messages"]
                        st.session_state.selected_brand = past["brand"]
                        st.rerun()

        st.divider()
        st.caption("Data in use")
        if st.session_state.loaded_files:
            for filename in list(st.session_state.loaded_files):
                col_name, col_remove = st.columns([5, 1])
                with col_name:
                    tag = " `sample`" if filename.startswith("[sample]") else ""
                    st.markdown(f"{filename}{tag}")
                with col_remove:
                    if st.button("×", key=f"remove_{filename}"):
                        vector_store.delete_by_source(CONNECT_COLLECTION, filename)
                        st.session_state.loaded_files.remove(filename)
                        st.rerun()
        else:
            st.caption("No data loaded - using general knowledge only.")

        with st.expander("+ Add your own · CSV, XLSX, PDF"):
            uploaded = st.file_uploader("Upload", type=["csv", "xlsx", "pdf"], accept_multiple_files=True, label_visibility="collapsed")
            if uploaded:
                for file in uploaded:
                    if file.name not in st.session_state.loaded_files:
                        ingest_file(file, file.name)
                st.rerun()

        if st.session_state.loaded_files and st.session_state.selected_brand != "none":
            if st.button("Reset to sample data", key="reset_data"):
                vector_store.clear(CONNECT_COLLECTION)
                st.session_state.structured_store.clear()
                st.session_state.loaded_files = []
                load_brand_preset(st.session_state.selected_brand)
                st.rerun()

        with st.expander("Brand context"):
            preset_context = st.session_state.brand_context
            brand = st.text_input("Brand", value=preset_context.get("brand", ""))
            product = st.text_input("Product", value=preset_context.get("product", ""))
            customer = st.text_input("Target customer", value=preset_context.get("customer", ""))
            category = st.text_input("Category", value=preset_context.get("category", ""))
            objective = st.text_input("Business objective", value=preset_context.get("objective", ""))
            st.caption("This copilot avoids medical/efficacy claims and flags anything that needs regulatory review.")

        show_comparison = st.checkbox("Also show generic comparison (extra API call)")

        if st.session_state.saved_ideas:
            st.divider()
            with st.expander(f"💾 Saved ideas ({len(st.session_state.saved_ideas)})"):
                for saved in st.session_state.saved_ideas:
                    st.markdown(f"**{saved.concept}**")
                    st.caption(saved.rationale[:140])
                if st.button("Clear saved ideas", use_container_width=True):
                    st.session_state.saved_ideas = []
                    st.rerun()

    def render_followup_chips(ideas, key_prefix: str):
        """Suggested follow-ups shown under a reply. Returns a follow-up query if one was clicked."""
        if not ideas:
            return None
        chips = []
        if len(ideas) >= 2:
            chips.append((f"Make idea 2 funnier", f'Make this idea funnier: "{ideas[1].concept}" - {ideas[1].rationale}'))
        chips.append(
            ("Write the full script for idea 1", f'Write a complete, detailed script for: "{ideas[0].concept}"')
        )
        chips.append(("Hindi versions", "Give me Hindi versions of these same ideas"))

        cols = st.columns(len(chips))
        follow_up = None
        for col, (label, prompt) in zip(cols, chips):
            with col:
                if st.button(label, key=f"chip_{key_prefix}_{label}", use_container_width=True):
                    follow_up = prompt
        return follow_up

    def render_assistant_message(message: dict, key_prefix: str):
        if message.get("error"):
            st.error(message["error"])
            return None
        if message.get("generic_ideas"):
            col_generic, col_data = st.columns(2)
            with col_generic:
                st.markdown("#### Generic (no data)")
                follow_up_a = render_idea_cards(message["generic_ideas"], key_prefix=f"{key_prefix}_generic")
            with col_data:
                st.markdown("#### With your data")
                follow_up_b = render_idea_cards(message["ideas"], key_prefix=f"{key_prefix}_data")
            return follow_up_a or follow_up_b
        follow_up = render_idea_cards(message["ideas"], key_prefix=key_prefix)
        return follow_up or render_followup_chips(message["ideas"], key_prefix=key_prefix)

    if st.session_state.selected_brand != "none":
        preset = BRAND_PRESETS[st.session_state.selected_brand]
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
    if not st.session_state.messages:
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
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                result = render_assistant_message(message, key_prefix=f"msg{msg_idx}")
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
        st.session_state.messages.append({"role": "user", "content": query})

        has_data = bool(st.session_state.loaded_files)
        mode = "connect" if has_data else "explore"
        structured_store = st.session_state.structured_store if has_data else StructuredStore()

        context = QueryContext(
            mode=mode,
            query=query,
            brand=brand,
            product=product,
            customer=customer,
            category=category,
            objective=objective,
        )

        error_message = None
        with st.spinner("Thinking..."):
            try:
                ideas = generate_ideas(
                    context=context,
                    vector_store=vector_store,
                    structured_store=structured_store,
                    llm_client=llm_client,
                )
            except Exception as e:
                error_message = describe_generation_error(e)
                ideas = []

            generic_ideas = []
            if show_comparison and has_data and not error_message:
                try:
                    with st.spinner("Generating generic comparison..."):
                        generic_ideas = generate_ideas(
                            context=QueryContext(mode="explore", query=query),
                            vector_store=vector_store,
                            structured_store=StructuredStore(),
                            llm_client=llm_client,
                            include_connect_data=False,
                        )
                except Exception:
                    pass  # comparison is a bonus, not critical - fail quietly rather than error the whole reply

        st.session_state.messages.append(
            {"role": "assistant", "ideas": ideas, "generic_ideas": generic_ideas, "error": error_message}
        )
        st.rerun()
