"""Tests for the pure-logic half of the brand restore flow (parse_restored_profile_json) - no
Streamlit runtime needed, since the widget wiring around it is just a thin wrapper."""
import json
from dataclasses import asdict

import pytest

from core.models import BrandProfile
from ui.brand_form import parse_restored_profile_json


def _downloaded_profile_bytes(**overrides) -> bytes:
    profile = BrandProfile(
        slug="joes-coffee",
        label="Coffee",
        brand_name="Joe's Coffee Co.",
        description="Third-wave coffee subscription.",
        context={"brand": "Joe's Coffee Co.", "product": "", "customer": "Coffee lovers", "category": "Coffee", "objective": "Retention"},
        voice_and_tone="Warm and knowledgeable.",
        positioning="Premium single-origin coffee.",
        personas="Coffee enthusiasts aged 25-45.",
        compliance_rules="Never claim health benefits from caffeine.",
        guide_filename="guide.md",
        is_custom=True,
    )
    data = asdict(profile)
    data.update(overrides)
    return json.dumps(data).encode("utf-8")


def test_parses_a_genuine_downloaded_profile():
    draft = parse_restored_profile_json(_downloaded_profile_bytes())

    assert draft["slug"] == "joes-coffee"
    assert draft["brand_name"] == "Joe's Coffee Co."
    assert draft["voice_and_tone"] == "Warm and knowledgeable."
    assert draft["positioning"] == "Premium single-origin coffee."
    assert draft["personas"] == "Coffee enthusiasts aged 25-45."
    assert draft["compliance_rules"] == "Never claim health benefits from caffeine."
    assert draft["target_customer"] == "Coffee lovers"
    assert draft["objective"] == "Retention"
    assert draft["data_files"] == []  # data files were never part of the download


def test_raises_value_error_on_invalid_json():
    with pytest.raises(ValueError):
        parse_restored_profile_json(b"not json at all")


def test_raises_value_error_when_a_required_field_is_missing():
    data = json.loads(_downloaded_profile_bytes())
    del data["compliance_rules"]
    with pytest.raises(ValueError):
        parse_restored_profile_json(json.dumps(data).encode("utf-8"))


def test_raises_value_error_on_non_utf8_bytes():
    with pytest.raises(ValueError):
        parse_restored_profile_json(b"\xff\xfe\x00\x01")


def test_falls_back_to_slugify_when_slug_is_missing():
    data = json.loads(_downloaded_profile_bytes())
    del data["slug"]
    draft = parse_restored_profile_json(json.dumps(data).encode("utf-8"))
    assert draft["slug"]  # some non-empty slug was derived from the brand name
