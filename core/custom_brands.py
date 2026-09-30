"""Persistence for marketer-created custom brands.

core/brand_presets.py is the built-in, curated catalogue (3 brands, shipped with the repo);
this module is the equivalent for brands a user creates at runtime through the "Add your
brand" flow, saved to data/brands/<slug>/ so they survive restarts and are available to every
future session, not just the one that created them. Already covered by the existing `data/*`
.gitignore rule - nothing extra needed there.

Layout per brand:
    data/brands/<slug>/
        profile.json      - the BrandProfile, as saved (source of truth for the picker)
        guide_text.txt     - the uploaded guide's extracted plain text, used to (re)build this
                             brand's Explore knowledge chunks (see core.knowledge.loader)
        files/
            <guide_filename>          - the original uploaded guide, kept for provenance
            <uploaded data filenames> - CSV/XLSX sample data, ingested into Connect at
                                        session-pick time exactly like a preset's sample_files
"""
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from core.brand_presets import BRAND_PRESETS
from core.models import BrandProfile

BRANDS_DIR = Path("data/brands")

# Slugs a custom brand must never collide with: the 3 presets, and the sentinels used
# elsewhere for "nothing picked yet" / "Other brand" (core.session.OTHER_BRAND_KEY, "general").
_RESERVED_SLUGS = set(BRAND_PRESETS) | {"general", "none"}


def slugify(name: str) -> str:
    """Turns a brand name into a filesystem-safe, collision-free slug. A name that collides
    with an existing slug (a preset, a reserved sentinel, or another saved custom brand) gets
    a numeric suffix rather than silently overwriting it."""
    base = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or "brand"
    existing = _RESERVED_SLUGS | set(list_custom_brands())
    slug = base
    suffix = 2
    while slug in existing:
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


def _brand_dir(slug: str) -> Path:
    return BRANDS_DIR / slug


def brand_files_dir(slug: str) -> Path:
    """Where a custom brand's uploaded files (guide + sample data) live - the custom-brand
    equivalent of the repo's sample_data/ directory for presets."""
    return _brand_dir(slug) / "files"


def save_custom_brand(
    profile: BrandProfile,
    guide_text: str,
    guide_filename: str,
    guide_bytes: bytes,
    data_files: list[tuple[str, bytes]],
) -> None:
    """Writes profile.json, guide_text.txt, and every uploaded file's raw bytes.

    `profile.sample_files` must already list just the data filenames, not the guide - only
    those get ingested into Connect when a session picks this brand (core.session.CopilotSession
    .load_brand()); the guide's content reaches the app instead via the shared Explore
    collection, chunked from guide_text.txt (see core.knowledge.loader.load_explore_knowledge).
    """
    files_dir = brand_files_dir(profile.slug)
    files_dir.mkdir(parents=True, exist_ok=True)

    (_brand_dir(profile.slug) / "profile.json").write_text(json.dumps(asdict(profile), indent=2))
    (_brand_dir(profile.slug) / "guide_text.txt").write_text(guide_text, encoding="utf-8")
    (files_dir / guide_filename).write_bytes(guide_bytes)
    for filename, content in data_files:
        (files_dir / filename).write_bytes(content)


def load_custom_brand(slug: str) -> Optional[BrandProfile]:
    path = _brand_dir(slug) / "profile.json"
    if not path.exists():
        return None
    try:
        return BrandProfile(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError):
        return None  # a corrupted save shouldn't be treated as "this brand exists"


def load_guide_text(slug: str) -> str:
    path = _brand_dir(slug) / "guide_text.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def list_custom_brands() -> dict[str, BrandProfile]:
    if not BRANDS_DIR.exists():
        return {}
    brands = {}
    for profile_path in sorted(BRANDS_DIR.glob("*/profile.json")):
        slug = profile_path.parent.name
        profile = load_custom_brand(slug)
        if profile is not None:
            brands[slug] = profile
    return brands


def delete_custom_brand(slug: str) -> None:
    brand_dir = _brand_dir(slug)
    if brand_dir.exists():
        shutil.rmtree(brand_dir)


def get_brand(slug: str) -> Optional[BrandProfile]:
    """The one lookup the rest of the app should use to resolve a brand key to its profile,
    regardless of whether it's a preset or a custom brand."""
    if slug in BRAND_PRESETS:
        return BRAND_PRESETS[slug]
    return load_custom_brand(slug)


def list_all_brands() -> dict[str, BrandProfile]:
    """Presets first (stable order), then custom brands - the one place the rest of the app
    should look to enumerate every pickable brand for the picker UI."""
    return {**BRAND_PRESETS, **list_custom_brands()}
