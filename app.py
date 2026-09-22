"""V0 Streamlit UI for the D2C Growth Copilot - brand picker + chat interface.

This file only handles UI state and wiring - all ingestion, retrieval, and
reasoning logic lives in core/.
"""
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.ingestion import get_parser
from core.knowledge.loader import load_explore_knowledge
from core.models import QueryContext
from core.reasoning.engine import generate_ideas
from core.reasoning.llm_client import QuotaExceededError, get_llm_client
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import CONNECT_COLLECTION, VectorStore

st.set_page_config(page_title="D2C Growth Copilot (V0)", layout="wide")

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

# The three categories this demo is deepest on. Picking one auto-loads that
# brand's own sample data + pre-fills its context - so users always know
# what's actually well-supported, rather than typing anything and guessing.
BRAND_PRESETS = {
    "sproutmix": {
        "label": "Kids Nutrition",
        "brand_name": "Sunny Sprout / SproutMix",
        "description": "Milk-mix nutrition brand for kids (SM2 / SM7 / SM13)",
        "sample_files": ["sproutmix_reviews.csv", "sproutmix_sales.xlsx"],
        "context": {
            "brand": "Sunny Sprout",
            "product": "SproutMix (SM2/SM7/SM13 milk mix)",
            "customer": "Parents of kids aged 2-13",
            "category": "Kids nutrition",
            "objective": "Acquisition and retention",
        },
    },
    "flexwear": {
        "label": "Fitness Apparel",
        "brand_name": "FlexWear",
        "description": "Activewear - Core/Power/Recovery leggings and joggers",
        "sample_files": ["flexwear_reviews.csv", "flexwear_sales.xlsx"],
        "context": {
            "brand": "FlexWear",
            "product": "Core/Power/Recovery activewear line",
            "customer": "Women 24-40, mixed fitness levels",
            "category": "Fitness apparel",
            "objective": "Acquisition and repeat purchase",
        },
    },
    "skincare": {
        "label": "Skincare & Personal Care",
        "brand_name": "GlowLabs",
        "description": "Vitamin C Face Wash, Niacinamide Serum, Sunscreen, plus a wider hair/body/gummies range",
        "sample_files": [
            "reviews.csv",
            "sales.xlsx",
            "personalcare_creative_tests.csv",
            "personalcare_narrative_angles.csv",
        ],
        "context": {
            "brand": "GlowLabs",
            "product": "Vitamin C Face Wash / Niacinamide Serum / Sunscreen SPF50",
            "customer": "Women 20-35 building a skincare routine",
            "category": "Skincare",
            "objective": "Acquisition",
        },
    },
}


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


def show_generation_error(exc: Exception) -> None:
    if isinstance(exc, QuotaExceededError):
        st.error(
            "Gemini's free-tier quota for this model is used up for now (a hard daily limit, "
            "not a blip) - either wait for it to reset, or set a different GEMINI_MODEL in .env."
        )
    else:
        st.error("The model is temporarily unavailable (this is on Google's side, not this app). Please try again in a moment.")


def render_idea_cards(ideas) -> None:
    for idea in ideas:
        with st.expander(idea.concept):
            st.markdown(f"**Rationale:** {idea.rationale}")
            st.markdown(f"**Recommended format:** {idea.recommended_format}")
            st.markdown(f"**Source/context used:** {idea.source_context}")
            if idea.script:
                st.markdown("**Script/brief:**")
                st.code(idea.script)


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

vector_store = get_vector_store()
llm_client = get_llm()

st.title("D2C Growth Copilot")

if st.session_state.selected_brand is None:
    st.subheader("Pick a brand to explore")
    st.caption(
        "This demo is deepest on these three categories - picking one loads its sample "
        "data and brand context automatically. Anything else falls back to general D2C "
        "knowledge only, so it's worth knowing which one you're in."
    )

    cols = st.columns(3)
    for col, (key, preset) in zip(cols, BRAND_PRESETS.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{preset['label']}**")
                st.caption(preset["brand_name"])
                st.write(preset["description"])
                if st.button("Explore this brand", key=f"pick_{key}", use_container_width=True):
                    load_brand_preset(key)
                    st.rerun()

    st.divider()
    if st.button("Skip - just chat without a specific brand"):
        st.session_state.selected_brand = "none"
        st.rerun()

else:
    with st.sidebar:
        if st.session_state.selected_brand != "none":
            preset = BRAND_PRESETS[st.session_state.selected_brand]
            st.success(f"Exploring: **{preset['label']}** ({preset['brand_name']})")
        else:
            st.info("Chatting without a specific brand - general D2C knowledge only.")

        if st.button("← Change brand", use_container_width=True):
            st.session_state.selected_brand = None
            st.rerun()

        st.divider()
        st.subheader("Your data")
        uploaded = st.file_uploader("Upload your own (CSV, XLSX, PDF)", type=["csv", "xlsx", "pdf"], accept_multiple_files=True)
        if uploaded:
            for file in uploaded:
                if file.name not in st.session_state.loaded_files:
                    ingest_file(file, file.name)
            st.rerun()

        if st.session_state.loaded_files:
            st.success("Loaded:\n" + "\n".join(f"- {f}" for f in st.session_state.loaded_files))
            if st.button("Reset data", use_container_width=True):
                vector_store.clear(CONNECT_COLLECTION)
                st.session_state.structured_store.clear()
                st.session_state.loaded_files = []
                st.rerun()

        with st.expander("Brand context"):
            preset_context = st.session_state.brand_context
            brand = st.text_input("Brand", value=preset_context.get("brand", ""))
            product = st.text_input("Product", value=preset_context.get("product", ""))
            customer = st.text_input("Target customer", value=preset_context.get("customer", ""))
            category = st.text_input("Category", value=preset_context.get("category", ""))
            objective = st.text_input("Business objective", value=preset_context.get("objective", ""))

        show_comparison = st.checkbox("Also show generic comparison (extra API call)")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_idea_cards(message["ideas"])
            else:
                st.markdown(message["content"])

    query = st.chat_input("What do you want to create?")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

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

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    ideas = generate_ideas(
                        context=context,
                        vector_store=vector_store,
                        structured_store=structured_store,
                        llm_client=llm_client,
                    )
            except Exception as e:
                show_generation_error(e)
                ideas = []

            generic_ideas = []
            if show_comparison and has_data:
                try:
                    with st.spinner("Generating generic comparison..."):
                        generic_ideas = generate_ideas(
                            context=QueryContext(mode="explore", query=query),
                            vector_store=vector_store,
                            structured_store=StructuredStore(),
                            llm_client=llm_client,
                            include_connect_data=False,
                        )
                except Exception as e:
                    show_generation_error(e)

            if show_comparison and has_data:
                col_generic, col_data = st.columns(2)
                with col_generic:
                    st.markdown("#### Generic (no data)")
                    render_idea_cards(generic_ideas)
                with col_data:
                    st.markdown("#### With your data")
                    render_idea_cards(ideas)
            else:
                render_idea_cards(ideas)

        st.session_state.messages.append({"role": "assistant", "ideas": ideas})
