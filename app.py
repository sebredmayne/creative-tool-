"""V0 Streamlit UI for the D2C Growth Copilot.

This file only handles UI state and wiring - all ingestion, retrieval, and
reasoning logic lives in core/.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.ingestion import get_parser
from core.knowledge.loader import load_explore_knowledge
from core.models import QueryContext
from core.reasoning.engine import generate_ideas
from core.reasoning.llm_client import get_llm_client
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import CONNECT_COLLECTION, EXPLORE_COLLECTION, VectorStore

st.set_page_config(page_title="D2C Growth Copilot (V0)", layout="wide")


@st.cache_resource
def get_vector_store() -> VectorStore:
    store = VectorStore()
    load_explore_knowledge(store)
    return store


@st.cache_resource
def get_llm():
    return get_llm_client()


def render_query_ui(mode: str, collection_name: str, context_fields: dict, structured_store: StructuredStore) -> None:
    """Shared chat-style input + output rendering, used by both Explore and Connect."""
    st.divider()
    query = st.text_input("What do you want to create?", key=f"query_{mode}")
    if st.button("Generate", key=f"generate_{mode}") and query:
        context = QueryContext(mode=mode, query=query, **context_fields)
        with st.spinner("Thinking..."):
            ideas = generate_ideas(
                context=context,
                collection_name=collection_name,
                vector_store=vector_store,
                structured_store=structured_store,
                llm_client=llm_client,
            )
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
if "mode" not in st.session_state:
    st.session_state.mode = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

vector_store = get_vector_store()
llm_client = get_llm()

st.title("D2C Growth Copilot — V0")

if st.session_state.mode is None:
    st.subheader("How would you like to get started?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Explore D2C", use_container_width=True):
            st.session_state.mode = "explore"
            st.rerun()
    with col2:
        if st.button("Connect Data", use_container_width=True):
            st.session_state.mode = "connect"
            st.rerun()

elif st.session_state.mode == "explore":
    st.subheader("Tell us about your brand")
    with st.form("explore_context"):
        brand = st.text_input("Brand")
        product = st.text_input("Product")
        customer = st.text_input("Target customer")
        category = st.text_input("Category")
        objective = st.text_input("Business objective")
        submitted = st.form_submit_button("Continue")

    if submitted:
        st.session_state.explore_context = {
            "brand": brand,
            "product": product,
            "customer": customer,
            "category": category,
            "objective": objective,
        }

    if st.session_state.get("explore_context"):
        # Explore mode never sees uploaded company data - pass a fresh, empty
        # StructuredStore so it can never leak in from a prior Connect session.
        render_query_ui(
            mode="explore",
            collection_name=EXPLORE_COLLECTION,
            context_fields=st.session_state.explore_context,
            structured_store=StructuredStore(),
        )

    if st.button("← Back", key="explore_back"):
        st.session_state.mode = None
        st.rerun()

elif st.session_state.mode == "connect":
    st.subheader("Upload your company data")
    uploaded = st.file_uploader("CSV, XLSX, or PDF", type=["csv", "xlsx", "pdf"], accept_multiple_files=True)

    if uploaded:
        for file in uploaded:
            if file.name in st.session_state.uploaded_files:
                continue
            parser = get_parser(file.name)
            if parser is None:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            parsed = parser.parse(file, file.name)
            ids = [f"{file.name}-{i}" for i in range(len(parsed.text_chunks))]
            metadatas = [{"source": file.name} for _ in parsed.text_chunks]
            vector_store.add_documents(CONNECT_COLLECTION, ids=ids, texts=parsed.text_chunks, metadatas=metadatas)

            if parsed.dataframes:
                for table_name, df in parsed.dataframes.items():
                    st.session_state.structured_store.add(table_name, df)

            st.session_state.uploaded_files.append(file.name)
            st.success(f"Ingested {file.name}")

    if st.session_state.uploaded_files:
        st.caption(f"Indexed files: {', '.join(st.session_state.uploaded_files)}")
        render_query_ui(
            mode="connect",
            collection_name=CONNECT_COLLECTION,
            context_fields={},
            structured_store=st.session_state.structured_store,
        )

    if st.button("Reset uploaded data"):
        vector_store.clear(CONNECT_COLLECTION)
        st.session_state.structured_store.clear()
        st.session_state.uploaded_files = []
        st.rerun()

    if st.button("← Back", key="connect_back"):
        st.session_state.mode = None
        st.rerun()
