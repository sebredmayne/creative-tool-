"""Tests for synthesize_trends's LLM response -> structured TrendsSummary parsing. Fake LLM
client, no real API calls."""
import pytest

from core.reasoning.trends_synthesis import TrendsSynthesisError, synthesize_trends


class _FakeClient:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def generate(self, system_prompt, user_prompt, response_schema=None):
        return self._response_text


_VALID_RESPONSE = (
    '{"overall_summary": "Most brands lean on convenience messaging.", '
    '"per_brand_notes": [{"brand_name": "Acme Milk Co", "note": "Focuses on quick prep."}], '
    '"patterns": ["Convenience angle", "Family-oriented visuals"], '
    '"gaps_or_opportunities": ["No brand addresses travel use cases."]}'
)

_SAMPLE_BRAND_SUMMARIES = [
    {"brand_name": "Acme Milk Co", "meta_ad_count": 2, "meta_samples": ["Quick prep!"], "google_ad_count": 0, "google_samples": []}
]


def test_parses_a_full_trends_summary_from_plain_json():
    summary = synthesize_trends("milk mix", _SAMPLE_BRAND_SUMMARIES, _FakeClient(_VALID_RESPONSE))
    assert summary.overall_summary == "Most brands lean on convenience messaging."
    assert summary.patterns == ["Convenience angle", "Family-oriented visuals"]
    assert summary.gaps_or_opportunities == ["No brand addresses travel use cases."]
    assert len(summary.per_brand_notes) == 1
    assert summary.per_brand_notes[0].brand_name == "Acme Milk Co"
    assert summary.per_brand_notes[0].note == "Focuses on quick prep."


def test_parses_from_markdown_fenced_json():
    fenced = f"```json\n{_VALID_RESPONSE}\n```"
    summary = synthesize_trends("milk mix", _SAMPLE_BRAND_SUMMARIES, _FakeClient(fenced))
    assert summary.overall_summary == "Most brands lean on convenience messaging."


def test_raises_trends_synthesis_error_on_invalid_json():
    with pytest.raises(TrendsSynthesisError):
        synthesize_trends("milk mix", _SAMPLE_BRAND_SUMMARIES, _FakeClient("not json at all"))


def test_raises_trends_synthesis_error_when_overall_summary_is_missing():
    incomplete = '{"per_brand_notes": [], "patterns": [], "gaps_or_opportunities": []}'
    with pytest.raises(TrendsSynthesisError):
        synthesize_trends("milk mix", _SAMPLE_BRAND_SUMMARIES, _FakeClient(incomplete))


def test_passes_category_and_brand_summaries_to_the_llm_client():
    captured = {}

    class _RecordingClient:
        def generate(self, system_prompt, user_prompt, response_schema=None):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            captured["response_schema"] = response_schema
            return _VALID_RESPONSE

    synthesize_trends("milk mix", _SAMPLE_BRAND_SUMMARIES, _RecordingClient())

    assert "milk mix" in captured["user_prompt"]
    assert "Acme Milk Co" in captured["user_prompt"]
    assert captured["response_schema"] is not None
    assert "never restate" in captured["system_prompt"] or "attribute" in captured["system_prompt"]
