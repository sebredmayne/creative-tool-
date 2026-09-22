# Project Plan

Living roadmap and decision log for the D2C Growth Copilot. See `README.md` for architecture and local setup.

## Vision

Two ways to give the AI context: **Explore** (no data required, curated knowledge) and **Connect** (upload/load real data), sharing one reasoning engine. V0 validates the mechanism; the real long-term differentiation isn't in single-shot generation quality - it's in three things not yet built: a performance feedback loop (generate -> observe results -> learn), genuine depth in the curated knowledge, and low-friction repeatability for non-technical users who won't write careful prompts themselves.

## Scope decision: three curated categories

Hand-authoring deep, compliance-aware knowledge for every possible D2C category doesn't scale. Settled on three representative examples instead of trying to generalize infinitely:

- **Kids nutrition** (SproutMix) - fictional, modeled on a real category's compliance structure
- **Fitness apparel** (FlexWear) - fictional
- **Skincare & personal care** (GlowLabs) - fictional, expanded to include a hair/body/supplement-gummies range

Each has brand voice + category-appropriate compliance rules (fixed, authoritative - never reinterpreted) and hook banks/benchmarks (illustrative, not fixed - see the dynamic-vs-fixed framing in `core/reasoning/prompts.py`). Anything outside these three falls back to general D2C knowledge only - the brand-picker UI in `app.py` exists specifically so users know which categories are actually well-supported instead of typing anything and getting ungrounded output.

## Completed

- V0 scaffold: Python + Streamlit + ChromaDB, ingestion (CSV/XLSX/PDF), separate Explore/Connect collections, a pandas-based structured-data store, a heuristic query router, live Gemini integration via the current `google-genai` SDK
- Chat-based UI (replaced the original form-first flow) with a 3-card brand picker as the landing gate
- Sample data for all three brands, including a 100+ row fictional creative-testing dataset (ad format / language / narrative-angle combinations) for GlowLabs
- Fixed: Explore and Connect were mutually exclusive - loading any data (even from an unrelated category) silently blocked all curated Explore knowledge. Now merged, with a relevance-floor cutoff on `VectorStore.query()` so irrelevant chunks get dropped instead of forced into every prompt
- Fixed: even after that merge, a growing Explore knowledge base could still crowd out genuinely relevant Connect data (tabular sample-data chunks tend to score slightly worse than well-written prose). `core/reasoning/engine.py::_merge_chunks()` now reserves a minimum number of retrieval slots for Connect data whenever it's loaded and has anything relevant
- Retry logic for Gemini's transient `503`s, plus a graceful UI error instead of a crash

## In progress

(nothing currently in progress)

## Deferred / future ideas

- Deepening the knowledge base further with more real frameworks, as they become available (fictionalized, per the confidentiality approach established for SproutMix/FlexWear/GlowLabs)
- GTM mode for pre-launch companies with no product history - needs a decision on whether it's a third top-level mode or a branch of Explore
- On-demand synthetic data generation (LLM-generated, category-specific) as a follow-up to the static sample library
- Rate limiting / cost controls if this ever moves beyond local personal use
- Deployment (currently local-only, no hosting)
