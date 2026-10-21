"""Tests for discover_top_brands's LLM response -> structured dict parsing. Fake LLM client,
no real API calls."""
import pytest

from core.brand_discovery import BrandDiscoveryError, discover_top_brands


class _FakeClient:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def generate(self, system_prompt, user_prompt, response_schema=None):
        return self._response_text


_VALID_RESPONSE = (
    '{"category_normalized": "milk mix", "brands": ['
    '{"brand_name": "Acme Milk Co", "domain": "acmemilk.com"}, '
    '{"brand_name": "Mystery Brand", "domain": null}]}'
)


def test_parses_brands_and_category_from_plain_json():
    result = discover_top_brands("milk mix", _FakeClient(_VALID_RESPONSE))
    assert result.category_normalized == "milk mix"
    assert [b.brand_name for b in result.brands] == ["Acme Milk Co", "Mystery Brand"]
    assert result.brands[0].domain == "acmemilk.com"
    assert result.brands[1].domain is None


def test_parses_brands_from_markdown_fenced_json():
    fenced = f"```json\n{_VALID_RESPONSE}\n```"
    result = discover_top_brands("milk mix", _FakeClient(fenced))
    assert len(result.brands) == 2


def test_raises_brand_discovery_error_on_invalid_json():
    with pytest.raises(BrandDiscoveryError):
        discover_top_brands("milk mix", _FakeClient("not json at all"))


def test_raises_brand_discovery_error_when_brand_name_is_missing():
    incomplete = '{"category_normalized": "milk mix", "brands": [{"domain": "acmemilk.com"}]}'
    with pytest.raises(BrandDiscoveryError):
        discover_top_brands("milk mix", _FakeClient(incomplete))


def test_passes_category_and_country_to_the_llm_client():
    captured = {}

    class _RecordingClient:
        def generate(self, system_prompt, user_prompt, response_schema=None):
            captured["user_prompt"] = user_prompt
            captured["response_schema"] = response_schema
            return _VALID_RESPONSE

    discover_top_brands("protein bars", _RecordingClient(), country="US", count=3)

    assert "protein bars" in captured["user_prompt"]
    assert "US" in captured["user_prompt"]
    assert "3" in captured["user_prompt"]
    assert captured["response_schema"] is not None
