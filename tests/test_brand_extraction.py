"""Tests for extract_brand_profile_fields's LLM response -> structured dict parsing. Uses a
fake LLM client, no real API calls."""
import pytest

from core.reasoning.brand_extraction import BrandExtractionError, extract_brand_profile_fields


class _FakeClient:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def generate(self, system_prompt, user_prompt, response_schema=None):
        return self._response_text


_VALID_RESPONSE = (
    '{"voice_and_tone": "Warm and direct.", "positioning": "Mid-premium activewear.", '
    '"personas": "Women 24-40.", "compliance_rules": "Never claim medical benefits."}'
)


def test_extracts_all_four_fields_from_plain_json():
    fields = extract_brand_profile_fields("some guide text", _FakeClient(_VALID_RESPONSE))
    assert fields == {
        "voice_and_tone": "Warm and direct.",
        "positioning": "Mid-premium activewear.",
        "personas": "Women 24-40.",
        "compliance_rules": "Never claim medical benefits.",
    }


def test_extracts_fields_from_markdown_fenced_json():
    fenced = f"```json\n{_VALID_RESPONSE}\n```"
    fields = extract_brand_profile_fields("some guide text", _FakeClient(fenced))
    assert fields["voice_and_tone"] == "Warm and direct."


def test_raises_extraction_error_on_invalid_json():
    with pytest.raises(BrandExtractionError):
        extract_brand_profile_fields("some guide text", _FakeClient("not json at all"))


def test_raises_extraction_error_when_a_field_is_missing():
    incomplete = '{"voice_and_tone": "Warm.", "positioning": "Premium.", "personas": "Adults."}'
    with pytest.raises(BrandExtractionError):
        extract_brand_profile_fields("some guide text", _FakeClient(incomplete))


def test_passes_the_guide_text_and_schema_to_the_llm_client():
    captured = {}

    class _RecordingClient:
        def generate(self, system_prompt, user_prompt, response_schema=None):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            captured["response_schema"] = response_schema
            return _VALID_RESPONSE

    extract_brand_profile_fields("A distinctive guide sentence about jaggery.", _RecordingClient())

    assert "A distinctive guide sentence about jaggery." in captured["user_prompt"]
    assert captured["response_schema"] is not None
    assert "compliance_rules" in captured["response_schema"]["properties"]
