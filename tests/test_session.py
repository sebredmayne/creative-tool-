"""Tests for CopilotSession - session-owned state and orchestration, with no Streamlit
dependency. Uses a real VectorStore/StructuredStore (temp-dir backed) and MockLLMClient, so
these run with zero network calls and without a running Streamlit app."""
import io
import json
import subprocess
import sys

import pytest

from core import custom_brands
from core.brand_presets import BRAND_PRESETS
from core.models import BrandProfile, CreativeIdea
from core.reasoning.llm_client import LLMClient, MockLLMClient
from core.retrieval.vector_store import VectorStore
from core.session import OTHER_BRAND_KEY, CopilotSession, UnsupportedFileTypeError


class _FailingLLMClient(LLMClient):
    def generate(self, system_prompt, user_prompt, response_schema=None):
        raise RuntimeError("simulated app-side failure")


class _QuotaExhaustedLLMClient(LLMClient):
    def generate(self, system_prompt, user_prompt, response_schema=None):
        from core.reasoning.llm_client import QuotaExceededError

        raise QuotaExceededError("simulated provider-side failure")


class _RecordingLLMClient(MockLLMClient):
    """Records every user_prompt it's given, in order, so a test can inspect what each turn
    actually sent - not just the fixed MockLLMClient output."""

    def __init__(self):
        self.user_prompts: list[str] = []

    def generate(self, system_prompt, user_prompt, response_schema=None):
        self.user_prompts.append(user_prompt)
        return super().generate(system_prompt, user_prompt)


@pytest.fixture
def session(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    return CopilotSession(vector_store, MockLLMClient())


def test_core_session_never_imports_streamlit():
    """The literal requirement: core/ must not import streamlit. Run in a fresh subprocess so
    it can't be contaminated by streamlit already being imported elsewhere in this test run."""
    result = subprocess.run(
        [sys.executable, "-c", "import core.session, sys; assert 'streamlit' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fresh_session_has_no_brand_selected_and_a_unique_connect_collection(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    session_a = CopilotSession(vector_store, MockLLMClient())
    session_b = CopilotSession(vector_store, MockLLMClient())

    assert session_a.selected_brand is None
    assert session_a.loaded_files == []
    assert session_a.messages == []
    assert session_a.connect_collection_name != session_b.connect_collection_name


def test_load_brand_ingests_sample_files_and_sets_context(session):
    session.load_brand("sproutmix")

    assert session.selected_brand == "sproutmix"
    assert session.brand_context == BRAND_PRESETS["sproutmix"].context
    assert len(session.loaded_files) == len(BRAND_PRESETS["sproutmix"].sample_files)
    assert all(f.startswith("[sample] ") for f in session.loaded_files)


def test_load_brand_clears_the_previous_brands_data_first(session):
    session.load_brand("sproutmix")
    sproutmix_files = set(session.loaded_files)

    session.load_brand("flexwear")

    assert session.selected_brand == "flexwear"
    assert not sproutmix_files & set(session.loaded_files)
    assert session.brand_context == BRAND_PRESETS["flexwear"].context


def test_set_other_brand_clears_data_and_uses_the_sentinel(session):
    session.load_brand("skincare")

    session.set_other_brand()

    assert session.selected_brand == OTHER_BRAND_KEY
    assert session.loaded_files == []
    assert session.brand_context == {}


def test_ingest_unsupported_file_type_raises_without_registering_it(session):
    with pytest.raises(UnsupportedFileTypeError):
        session.ingest(io.BytesIO(b"whatever"), "notes.docx")
    assert "notes.docx" not in session.loaded_files


def test_remove_file_drops_it_from_loaded_files(session):
    session.load_brand("sproutmix")
    filename = session.loaded_files[0]

    session.remove_file(filename)

    assert filename not in session.loaded_files


def test_reset_to_sample_drops_any_extra_uploads(session):
    session.load_brand("sproutmix")
    original_files = set(session.loaded_files)

    session.ingest(io.BytesIO(b"a,b\n1,2\n"), "extra.csv")
    assert "extra.csv" in session.loaded_files

    session.reset_to_sample()

    assert set(session.loaded_files) == original_files


def test_ask_returns_ideas_and_records_chat_history(session):
    result = session.ask("give me a reel script", brand_context={})

    assert len(result.ideas) >= 1
    assert result.error is None
    assert session.messages[-2] == {"role": "user", "content": "give me a reel script"}
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["ideas"] == result.ideas


def test_ask_maps_an_app_side_failure_to_an_error_message_instead_of_raising(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    failing_session = CopilotSession(vector_store, _FailingLLMClient())

    result = failing_session.ask("give me a reel script", brand_context={})

    assert result.ideas == []
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert result.error_is_provider_side is False
    assert failing_session.messages[-1]["error"] == result.error


def test_ask_flags_a_provider_side_failure_as_such(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    quota_session = CopilotSession(vector_store, _QuotaExhaustedLLMClient())

    result = quota_session.ask("give me a reel script", brand_context={})

    assert result.error is not None
    assert result.error_is_provider_side is True


def test_ask_only_runs_generic_comparison_when_connect_data_is_loaded(session):
    result = session.ask("give me a reel script", brand_context={}, compare_generic=True)
    assert result.generic_ideas == []  # no Connect data loaded, so no comparison to run

    session.load_brand("sproutmix")
    result = session.ask("give me a reel script", brand_context={}, compare_generic=True)
    assert len(result.generic_ideas) >= 1


def test_save_idea_is_idempotent(session):
    result = session.ask("give me a reel script", brand_context={})
    idea = result.ideas[0]

    session.save_idea(idea)
    session.save_idea(idea)

    assert session.saved_ideas == [idea]


def test_clear_saved_ideas(session):
    result = session.ask("give me a reel script", brand_context={})
    session.save_idea(result.ideas[0])

    session.clear_saved_ideas()

    assert session.saved_ideas == []


def test_new_chat_archives_the_current_conversation(session):
    session.ask("give me a reel script", brand_context={})

    session.new_chat()

    assert session.messages == []
    assert len(session.past_chats) == 1
    assert session.past_chats[0]["title"] == "give me a reel script"


def test_restore_chat_brings_back_its_messages_and_brand(session):
    session.load_brand("sproutmix")
    session.ask("give me a reel script", brand_context={})
    session.new_chat()
    archived_messages = session.past_chats[0]["messages"]

    session.load_brand("flexwear")  # switch away before restoring

    session.restore_chat(0)

    assert session.messages == archived_messages
    assert session.selected_brand == "sproutmix"


def test_first_message_prompt_has_no_previous_ideas_section(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    llm_client = _RecordingLLMClient()
    session = CopilotSession(vector_store, llm_client)

    session.ask("give me a reel script", brand_context={})

    assert "previous turn" not in llm_client.user_prompts[0].lower()


def test_followup_prompt_includes_the_previous_turns_ideas(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    llm_client = _RecordingLLMClient()
    session = CopilotSession(vector_store, llm_client)

    first_result = session.ask("give me a reel script", brand_context={})
    session.ask("make it funnier", brand_context={})

    assert len(llm_client.user_prompts) == 2
    followup_prompt = llm_client.user_prompts[1]
    assert "previous turn" in followup_prompt.lower()
    assert first_result.ideas[0].concept in followup_prompt


def test_load_brand_works_for_a_custom_brand_too(session, tmp_path, monkeypatch):
    monkeypatch.setattr(custom_brands, "BRANDS_DIR", tmp_path / "brands")
    profile = BrandProfile(
        slug="joes-coffee",
        label="Coffee",
        brand_name="Joe's Coffee Co.",
        description="Third-wave coffee subscription.",
        context={"brand": "Joe's Coffee Co.", "product": "Coffee", "customer": "", "category": "", "objective": ""},
        compliance_rules="Never claim health benefits from caffeine.",
        sample_files=["reviews.csv"],
        guide_filename="guide.md",
        is_custom=True,
    )
    custom_brands.save_custom_brand(
        profile,
        guide_text="Guide text.",
        guide_filename="guide.md",
        guide_bytes=b"Guide text.",
        data_files=[("reviews.csv", b"review\nGreat coffee, fast shipping.\n")],
    )

    session.load_brand("joes-coffee")

    assert session.selected_brand == "joes-coffee"
    assert session.brand_context == profile.context
    assert session.loaded_files == ["[sample] reviews.csv"]


def test_generate_edit_brief_uses_the_selected_brands_name_and_compliance_rules(session):
    captured = {}

    class _RecordingClient:
        def generate(self, system_prompt, user_prompt, response_schema=None):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            return json.dumps(
                {
                    "aspect_ratio": "9:16",
                    "duration_seconds": 5,
                    "alternative_hooks": [],
                    "scenes": [
                        {
                            "timestamp_range": "0:00-0:05",
                            "beat_type": "hook",
                            "voiceover_or_onscreen_text": "x",
                            "visual_description": "y",
                        }
                    ],
                    "music_mood_note": "",
                }
            )

    session._llm_client = _RecordingClient()
    session.load_brand("sproutmix")
    idea = CreativeIdea(
        concept="Test idea", rationale="r", recommended_format="Instagram Reel", source_context="s"
    )

    result = session.generate_edit_brief(idea)

    assert result.error is None
    assert result.brief.brand == "Sunny Sprout"
    assert "medical advice" in captured["user_prompt"].lower()  # SproutMix's compliance disclaimer


def test_generate_edit_brief_returns_an_error_result_instead_of_raising(tmp_path):
    class _BrokenClient(LLMClient):
        def generate(self, system_prompt, user_prompt, response_schema=None):
            return "not valid json"

    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    broken_session = CopilotSession(vector_store, _BrokenClient())
    idea = CreativeIdea(concept="c", rationale="r", recommended_format="Reel", source_context="s")

    result = broken_session.generate_edit_brief(idea)

    assert result.brief is None
    assert result.error is not None


def test_ask_blocks_once_the_generation_limit_is_reached(session, monkeypatch):
    from core import session as session_module

    monkeypatch.setattr(session_module, "MAX_GENERATIONS_PER_SESSION", 2)

    session.ask("first", brand_context={})
    session.ask("second", brand_context={})
    assert session.generation_count == 2

    result = session.ask("third - should be blocked", brand_context={})

    assert result.ideas == []
    assert result.error is not None
    assert "caps each session" in result.error
    assert session.generation_count == 2  # the blocked attempt didn't consume another slot
    assert session.messages[-1]["error"] == result.error


def test_generate_edit_brief_blocks_once_the_generation_limit_is_reached(session, monkeypatch):
    from core import session as session_module

    monkeypatch.setattr(session_module, "MAX_GENERATIONS_PER_SESSION", 0)
    idea = CreativeIdea(concept="c", rationale="r", recommended_format="Reel", source_context="s")

    result = session.generate_edit_brief(idea)

    assert result.brief is None
    assert "caps each session" in result.error
    assert session.generation_count == 0


def test_generation_count_increments_once_per_ask_and_once_more_for_comparison(session):
    session.ask("a query", brand_context={}, compare_generic=False)
    assert session.generation_count == 1

    session.load_brand("sproutmix")  # compare_generic only runs when Connect data is loaded
    session.ask("another query", brand_context={}, compare_generic=True)
    assert session.generation_count == 3  # +1 main call, +1 comparison call
