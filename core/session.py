"""One user's session: brand selection, Connect data, chat history, and request
orchestration - everything app.py used to hold in st.session_state and inline logic.

Plain Python, no Streamlit dependency, so it's usable and testable outside a running app.
app.py should do nothing more than construct one CopilotSession per browser session, call its
methods from widget callbacks, and render whatever it returns.
"""
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core import custom_brands
from core.ingestion import get_parser
from core.knowledge.loader import load_compliance_rules
from core.models import CreativeIdea, EditBrief, QueryContext
from core.reasoning.edit_brief import EditBriefError
from core.reasoning.edit_brief import generate_edit_brief as _generate_edit_brief
from core.reasoning.engine import generate_ideas
from core.reasoning.llm_client import ChainExhaustedError, LLMClient, QuotaExceededError
from core.retrieval.structured_store import StructuredStore
from core.retrieval.vector_store import VectorStore

logger = logging.getLogger("d2c_growth_copilot")

SAMPLE_DATA_DIR = Path(__file__).parent.parent / "sample_data"

# Sentinel selected_brand value for a session with no specific brand ("Other brand" in the
# UI) - distinct from None, which means "nothing picked yet at all".
OTHER_BRAND_KEY = "none"

# Per-session cap on real generation attempts (ask() + generate_edit_brief() combined) - quota
# protection for a public demo running on one shared free-tier key. Overridable via env/secrets
# for a private deployment that doesn't need it as tight (or at all - set it very high).
MAX_GENERATIONS_PER_SESSION = int(os.getenv("MAX_GENERATIONS_PER_SESSION", "20"))


class UnsupportedFileTypeError(Exception):
    """Raised by CopilotSession.ingest() when there's no parser for a file's extension. Core
    doesn't know about warnings/toasts, so it's up to the caller (the UI) to decide how to
    surface this rather than have core reach for st.warning() itself."""


@dataclass
class AskResult:
    """What CopilotSession.ask() hands back for one turn."""

    ideas: list[CreativeIdea] = field(default_factory=list)
    generic_ideas: list[CreativeIdea] = field(default_factory=list)
    error: Optional[str] = None
    # True for a transient/external failure (every model's quota exhausted, overloaded, timed
    # out) - not a bug in this app, and something a caller (e.g. the eval harness) may want to
    # treat as "couldn't get a result" rather than "produced a bad one".
    error_is_provider_side: bool = False


def classify_generation_error(exc: Exception) -> tuple[bool, str]:
    """Returns (is_provider_side, message) for a generation failure. Only genuinely
    provider-side failures should be described that way - anything else (a bug here, a parse
    error, a local network issue) should say that plainly instead of misleadingly blaming the
    provider. Imports are lazy so this module still works with only MockLLMClient installed."""
    import json

    import httpx
    from google.genai import errors as genai_errors

    if isinstance(exc, ChainExhaustedError):
        attempted = ", ".join(label for label, _ in exc.attempts)
        return True, (
            f"Every configured free-tier model failed ({attempted}). This usually means "
            "several models are quota-exhausted or overloaded at the same time - wait a "
            "moment and try again, or add another provider (GROQ_API_KEY or OLLAMA_MODEL) "
            "in .env for more coverage. See logs/app.log for what each model actually returned."
        )
    if isinstance(exc, QuotaExceededError):
        return True, (
            "This model's free-tier quota is used up for now (a hard daily limit, not a "
            "blip) - either wait for it to reset, or add more models to GEMINI_MODELS in .env."
        )
    if isinstance(exc, genai_errors.ServerError):
        return True, "The model is temporarily unavailable (this is on Google's side, not this app). Please try again in a moment."
    if isinstance(exc, httpx.TimeoutException):
        return True, "The request to the model timed out. Please try again."
    if isinstance(exc, json.JSONDecodeError):
        return False, "The model's response couldn't be parsed. Please try again or rephrase the question."
    if isinstance(exc, EditBriefError):
        return False, f"The model's edit brief wasn't valid ({exc}). Try again - a re-generation often resolves it."
    return False, (
        f"Something went wrong in this app, not the model provider's API ({type(exc).__name__}). "
        "This has been logged - check logs/app.log for details."
    )


def describe_generation_error(exc: Exception) -> str:
    """Just the message half of classify_generation_error - kept as its own function since
    that's what most callers (the UI) actually want."""
    return classify_generation_error(exc)[1]


def _generation_limit_message() -> str:
    return (
        f"This demo caps each session at {MAX_GENERATIONS_PER_SESSION} generations, to keep "
        "one shared free-tier key usable for everyone sharing this link - refresh the page to "
        "start a new session, or run this locally with your own key for unlimited use."
    )


@dataclass
class EditBriefResult:
    """What CopilotSession.generate_edit_brief() hands back."""

    brief: Optional[EditBrief] = None
    error: Optional[str] = None


class CopilotSession:
    """One browser session's worth of state, constructed once and stored in
    st.session_state by app.py. `vector_store` and `llm_client` are shared singletons
    (cached at the process level) passed in rather than constructed here."""

    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self._vector_store = vector_store
        self._llm_client = llm_client

        # Unique per session so one session's uploads/sample data are never visible to
        # another's queries - this collection name is what actually enforces that isolation.
        self.connect_collection_name = f"connect_{uuid.uuid4().hex}"
        self.structured_store = StructuredStore()
        self.loaded_files: list[str] = []
        self.messages: list[dict] = []
        self.selected_brand: Optional[str] = None
        self.brand_context: dict = {}
        self.saved_ideas: list[CreativeIdea] = []
        self.past_chats: list[dict] = []
        # Counts every real generation attempt (ask() + generate_edit_brief()), regardless of
        # success/failure - a public demo link uses one shared free-tier key, so this caps how
        # much of it one browser session can spend, not just how much quota is left overall.
        self.generation_count: int = 0

    # --- brand selection ---------------------------------------------------------------

    def load_brand(self, key: str) -> None:
        profile = custom_brands.get_brand(key)
        if profile is None:
            raise ValueError(f"Unknown brand: {key!r}")

        self._clear_connect_data()
        data_dir = custom_brands.brand_files_dir(key) if profile.is_custom else SAMPLE_DATA_DIR
        for filename in profile.sample_files:
            sample_name = f"[sample] {filename}"
            with open(data_dir / filename, "rb") as f:
                self.ingest(f, sample_name)
        self.selected_brand = key
        self.brand_context = profile.context

    def set_other_brand(self) -> None:
        self._clear_connect_data()
        self.selected_brand = OTHER_BRAND_KEY
        self.brand_context = {}

    def reset_to_sample(self) -> None:
        """Clears whatever's loaded and reloads the current brand's sample data - only valid
        once a real brand (not "Other brand") has been selected."""
        self.load_brand(self.selected_brand)

    def _clear_connect_data(self) -> None:
        """Wipes this session's Connect data. Used whenever the brand changes, since a
        session's connect_<uuid> collection is reused for its whole lifetime rather than
        rotated, so it must be explicitly emptied before loading a different brand's data."""
        self._vector_store.clear(self.connect_collection_name)
        self.structured_store.clear()
        self.loaded_files = []

    # --- data management -----------------------------------------------------------------

    def ingest(self, file, filename: str) -> None:
        """Parse a file and add it to this session's Connect vector store + structured
        store. Shared by real uploads and brand-preset sample data."""
        parser = get_parser(filename)
        if parser is None:
            raise UnsupportedFileTypeError(filename)

        parsed = parser.parse(file, filename)
        ids = [f"{filename}-{i}" for i in range(len(parsed.text_chunks))]
        metadatas = [{"source": filename} for _ in parsed.text_chunks]
        self._vector_store.add_documents(
            self.connect_collection_name, ids=ids, texts=parsed.text_chunks, metadatas=metadatas
        )

        if parsed.dataframes:
            for table_name, df in parsed.dataframes.items():
                self.structured_store.add(table_name, df)

        self.loaded_files.append(filename)

    def remove_file(self, filename: str) -> None:
        self._vector_store.delete_by_source(self.connect_collection_name, filename)
        self.loaded_files.remove(filename)

    # --- chat ---------------------------------------------------------------------------

    def ask(self, query: str, brand_context: dict, compare_generic: bool = False) -> AskResult:
        """Runs one full turn: records the user message, generates ideas (scoped to this
        session's brand/Connect data), optionally runs a Connect-free comparison, records the
        assistant reply, and returns it. `brand_context` is the live brand/product/customer/
        category/objective values from the UI, not necessarily self.brand_context (which is
        only the preset's initial prefill)."""
        # Captured before appending this turn's user message, so these reflect the *prior*
        # turn - a bare follow-up ("Hindi versions of these") otherwise reaches generate_ideas
        # with no idea what "these" refers to.
        previous_query = next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), None)
        previous_ideas = next((m["ideas"] for m in reversed(self.messages) if m["role"] == "assistant"), None)

        self.messages.append({"role": "user", "content": query})

        if self.generation_count >= MAX_GENERATIONS_PER_SESSION:
            result = AskResult(error=_generation_limit_message())
            self.messages.append(
                {"role": "assistant", "ideas": [], "generic_ideas": [], "error": result.error}
            )
            return result

        has_data = bool(self.loaded_files)
        mode = "connect" if has_data else "explore"
        structured_store = self.structured_store if has_data else StructuredStore()
        explore_brand = self.selected_brand if self.selected_brand != OTHER_BRAND_KEY else "general"

        context = QueryContext(mode=mode, query=query, **brand_context)

        error_message = None
        error_is_provider_side = False
        ideas: list[CreativeIdea] = []
        self.generation_count += 1
        try:
            ideas = generate_ideas(
                context=context,
                vector_store=self._vector_store,
                structured_store=structured_store,
                llm_client=self._llm_client,
                connect_collection_name=self.connect_collection_name,
                explore_brand=explore_brand,
                previous_query=previous_query,
                previous_ideas=previous_ideas,
            )
        except Exception as e:
            logger.exception("generate_ideas failed for query: %r", query)
            error_is_provider_side, error_message = classify_generation_error(e)

        generic_ideas: list[CreativeIdea] = []
        if compare_generic and has_data and not error_message:
            self.generation_count += 1
            try:
                generic_ideas = generate_ideas(
                    context=QueryContext(mode="explore", query=query),
                    vector_store=self._vector_store,
                    structured_store=StructuredStore(),
                    llm_client=self._llm_client,
                    include_connect_data=False,
                    explore_brand=explore_brand,
                    previous_query=previous_query,
                    previous_ideas=previous_ideas,
                )
            except Exception:
                # Comparison is a bonus, not critical - fail quietly rather than error the
                # whole reply, but still log it so failures aren't invisible.
                logger.exception("generic comparison generate_ideas failed for query: %r", query)

        result = AskResult(
            ideas=ideas, generic_ideas=generic_ideas, error=error_message, error_is_provider_side=error_is_provider_side
        )
        self.messages.append(
            {"role": "assistant", "ideas": result.ideas, "generic_ideas": result.generic_ideas, "error": result.error}
        )
        return result

    # --- saved ideas / past chats ---------------------------------------------------------

    def save_idea(self, idea: CreativeIdea) -> None:
        if idea not in self.saved_ideas:
            self.saved_ideas.append(idea)

    def clear_saved_ideas(self) -> None:
        self.saved_ideas = []

    def new_chat(self) -> None:
        """Archives the current conversation (if any) into past_chats, then starts a fresh
        one - "+ New chat" in the UI."""
        if self.messages:
            first_user_msg = next((m["content"] for m in self.messages if m["role"] == "user"), "Chat")
            self.past_chats.append(
                {"title": first_user_msg[:40], "brand": self.selected_brand, "messages": self.messages}
            )
        self.messages = []

    def restore_chat(self, index: int) -> None:
        past = self.past_chats[index]
        self.messages = past["messages"]
        self.selected_brand = past["brand"]

    # --- edit briefs ----------------------------------------------------------------------

    def generate_edit_brief(self, idea: CreativeIdea) -> EditBriefResult:
        """Turns one already-generated idea into a production-ready scene-by-scene brief -
        one LLM call, using the idea's own concept/rationale/script and this session's brand
        compliance rules. Never regenerates the idea itself."""
        if self.generation_count >= MAX_GENERATIONS_PER_SESSION:
            return EditBriefResult(error=_generation_limit_message())

        # Normalizes both "no brand selected yet" (None) and "Other brand" (OTHER_BRAND_KEY) to
        # "general" - unlike ask()'s use of the same pattern, custom_brands.get_brand() doesn't
        # accept None, so this one has to be explicit about both cases rather than just the one.
        explore_brand = self.selected_brand if self.selected_brand not in (None, OTHER_BRAND_KEY) else "general"
        compliance_rules = load_compliance_rules(explore_brand)
        brand_profile = custom_brands.get_brand(explore_brand) if explore_brand != "general" else None
        brand_name = brand_profile.brand_name.split(" / ")[0] if brand_profile else "General D2C brand"

        self.generation_count += 1
        try:
            brief = _generate_edit_brief(idea, brand_name, compliance_rules, self._llm_client)
            return EditBriefResult(brief=brief)
        except Exception as e:
            logger.exception("generate_edit_brief failed for idea: %r", idea.concept)
            _, message = classify_generation_error(e)
            return EditBriefResult(error=message)
