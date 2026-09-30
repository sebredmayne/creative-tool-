"""Loads the curated Explore-mode D2C knowledge base (markdown files) into the vector store."""
import hashlib
from pathlib import Path
from typing import Optional

from core.brand_presets import BRAND_PRESETS
from core.custom_brands import list_custom_brands, load_custom_brand, load_guide_text
from core.ingestion.chunking import chunk_text
from core.retrieval.vector_store import EXPLORE_COLLECTION, VectorStore

KNOWLEDGE_DIR = Path(__file__).parent / "explore"

# A short, hardcoded fallback for brand-less sessions ("Other brand" in the UI) - deliberately
# not a knowledge file, since it's meant to be the minimum safe default, not a deep-dive.
GENERAL_COMPLIANCE_RULES = (
    "- Never state or imply a medical or efficacy claim (curing, treating, preventing, or "
    "guaranteeing any health/medical outcome), regardless of product category.\n"
    "- Flag anything touching health, medical, or efficacy claims for human/regulatory review "
    "(set needs_review=true) rather than deciding on your own that it's fine.\n"
)


def _brand_for_file(stem: str) -> str:
    """Derives a chunk's brand from its filename prefix (e.g. "sproutmix_..." -> "sproutmix"),
    so Explore retrieval can be scoped to the brand actually being discussed (see
    core.reasoning.engine.generate_ideas) instead of mixing every brand's knowledge together."""
    for brand_key in BRAND_PRESETS:
        if stem.startswith(f"{brand_key}_"):
            return brand_key
    return "general"


def _explore_source_files() -> list[Path]:
    """Every knowledge file that actually gets embedded into the Explore collection - i.e. not
    the `*_compliance.md` files, which are loaded deterministically (see load_compliance_rules)
    and deliberately excluded here so they aren't also present as a second, retrieval-only copy."""
    return [path for path in sorted(KNOWLEDGE_DIR.glob("*.md")) if not path.stem.endswith("_compliance")]


def _current_knowledge_hash() -> str:
    """A single hash over every Explore source file's name and content, plus every custom
    brand's guide text - used to detect whether the collection is stale. Not a security hash,
    just a fast "did anything change" check. Custom brands are included so creating, editing,
    or deleting one triggers a rebuild exactly like editing a static .md file does."""
    hasher = hashlib.sha256()
    for path in _explore_source_files():
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    for slug in sorted(list_custom_brands()):
        hasher.update(slug.encode("utf-8"))
        hasher.update(load_guide_text(slug).encode("utf-8"))
    return hasher.hexdigest()


def load_explore_knowledge(vector_store: VectorStore) -> None:
    """(Re)populate the Explore collection from local markdown files and every saved custom
    brand's guide - but only when something's actually changed since the last build. A hash of
    all of it is stored on the collection itself (Chroma collection metadata), so an unchanged
    knowledge base skips the embedding cost entirely on every process start, while an edited
    .md file or a newly created/edited/deleted custom brand is picked up on the very next
    start instead of silently staying stale until someone notices and manually clears
    data/chroma - which is exactly what used to happen, since this previously only checked "is
    the collection non-empty", not "does it match what's on disk right now".
    """
    current_hash = _current_knowledge_hash()
    if vector_store.get_collection_metadata(EXPLORE_COLLECTION).get("knowledge_hash") == current_hash:
        return

    vector_store.clear(EXPLORE_COLLECTION)

    ids, texts, metadatas = [], [], []
    for path in _explore_source_files():
        content = path.read_text(encoding="utf-8")
        brand = _brand_for_file(path.stem)
        for i, chunk in enumerate(chunk_text(content, chunk_size=150, overlap=30)):
            ids.append(f"{path.stem}-{i}")
            texts.append(chunk)
            metadatas.append({"source": path.name, "brand": brand})

    for slug in sorted(list_custom_brands()):
        guide_text = load_guide_text(slug)
        for i, chunk in enumerate(chunk_text(guide_text, chunk_size=150, overlap=30)):
            ids.append(f"custom-{slug}-{i}")
            texts.append(chunk)
            metadatas.append({"source": f"{slug}_brand_guide", "brand": slug})

    vector_store.add_documents(EXPLORE_COLLECTION, ids=ids, texts=texts, metadatas=metadatas)
    vector_store.set_collection_metadata(EXPLORE_COLLECTION, {"knowledge_hash": current_hash})


def load_compliance_rules(brand_key: Optional[str]) -> str:
    """Deterministically loads one brand's compliance/claim rules as plain text, bypassing
    retrieval entirely - these are non-negotiable, so they must reach the system prompt every
    time, not only when vector search happens to rank them in the top_k.

    Checks a saved custom brand first (its compliance_rules field, user-confirmed at creation
    time), then a preset's static `<brand>_compliance.md` file, falling back to
    GENERAL_COMPLIANCE_RULES for a brand-less session ("Other brand") or a brand with neither.
    """
    if brand_key and brand_key != "general":
        custom_profile = load_custom_brand(brand_key)
        if custom_profile is not None:
            return custom_profile.compliance_rules or GENERAL_COMPLIANCE_RULES
        path = KNOWLEDGE_DIR / f"{brand_key}_compliance.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
    return GENERAL_COMPLIANCE_RULES
