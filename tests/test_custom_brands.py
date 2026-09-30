"""Tests for custom brand persistence: save/load round trip, listing, deletion, and slug
collision handling. Uses a temp BRANDS_DIR (monkeypatched) so these never touch the real
data/brands/ directory."""
import pytest

from core import custom_brands
from core.models import BrandProfile


@pytest.fixture(autouse=True)
def temp_brands_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_brands, "BRANDS_DIR", tmp_path / "brands")


def _profile(slug="joes-coffee", **overrides) -> BrandProfile:
    defaults = dict(
        slug=slug,
        label="Coffee",
        brand_name="Joe's Coffee Co.",
        description="Third-wave coffee subscription.",
        context={"brand": "Joe's Coffee Co.", "product": "Coffee subscription", "customer": "", "category": "Coffee", "objective": "Retention"},
        voice_and_tone="Friendly and knowledgeable.",
        positioning="Premium single-origin beans.",
        personas="Coffee enthusiasts aged 25-45.",
        compliance_rules="Never claim health benefits from caffeine.",
        sample_files=["reviews.csv"],
        example_prompts=[],
        guide_filename="brand_guide.md",
        is_custom=True,
    )
    defaults.update(overrides)
    return BrandProfile(**defaults)


def test_save_and_load_round_trip():
    profile = _profile()
    custom_brands.save_custom_brand(
        profile,
        guide_text="Full guide text here.",
        guide_filename="brand_guide.md",
        guide_bytes=b"Full guide text here.",
        data_files=[("reviews.csv", b"a,b\n1,2\n")],
    )

    loaded = custom_brands.load_custom_brand("joes-coffee")

    assert loaded == profile
    assert custom_brands.load_guide_text("joes-coffee") == "Full guide text here."
    assert (custom_brands.brand_files_dir("joes-coffee") / "reviews.csv").read_bytes() == b"a,b\n1,2\n"
    assert (custom_brands.brand_files_dir("joes-coffee") / "brand_guide.md").exists()


def test_load_nonexistent_brand_returns_none():
    assert custom_brands.load_custom_brand("does-not-exist") is None


def test_list_custom_brands_returns_every_saved_brand():
    custom_brands.save_custom_brand(_profile(slug="brand-a"), "guide a", "g.md", b"guide a", [])
    custom_brands.save_custom_brand(_profile(slug="brand-b"), "guide b", "g.md", b"guide b", [])

    brands = custom_brands.list_custom_brands()

    assert set(brands) == {"brand-a", "brand-b"}


def test_delete_custom_brand_removes_it_from_the_listing():
    custom_brands.save_custom_brand(_profile(), "guide text", "g.md", b"guide text", [])
    assert "joes-coffee" in custom_brands.list_custom_brands()

    custom_brands.delete_custom_brand("joes-coffee")

    assert "joes-coffee" not in custom_brands.list_custom_brands()
    assert custom_brands.load_custom_brand("joes-coffee") is None


def test_delete_nonexistent_brand_does_not_raise():
    custom_brands.delete_custom_brand("never-existed")  # must not raise


def test_get_brand_finds_both_presets_and_custom_brands():
    custom_brands.save_custom_brand(_profile(), "guide text", "g.md", b"guide text", [])

    assert custom_brands.get_brand("sproutmix") is not None
    assert custom_brands.get_brand("sproutmix").is_custom is False
    assert custom_brands.get_brand("joes-coffee").is_custom is True
    assert custom_brands.get_brand("no-such-brand") is None


def test_list_all_brands_includes_presets_and_custom():
    custom_brands.save_custom_brand(_profile(), "guide text", "g.md", b"guide text", [])

    all_brands = custom_brands.list_all_brands()

    assert {"sproutmix", "skincare", "flexwear", "joes-coffee"} <= set(all_brands)


def test_slugify_avoids_colliding_with_presets():
    assert custom_brands.slugify("SproutMix") != "sproutmix"


def test_slugify_avoids_colliding_with_an_existing_custom_brand():
    custom_brands.save_custom_brand(_profile(slug="joes-coffee"), "guide text", "g.md", b"guide text", [])

    assert custom_brands.slugify("Joe's Coffee") != "joes-coffee"
