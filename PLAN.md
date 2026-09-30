# Project Plan

Living roadmap and decision log for the D2C Growth Copilot. See `README.md` for architecture and local setup.

## Vision

Two ways to give the AI context: **Explore** (no data required, curated knowledge) and **Connect** (upload/load real data), sharing one reasoning engine. V0 validates the mechanism; the real long-term differentiation isn't in single-shot generation quality - it's in three things not yet built: a performance feedback loop (generate -> observe results -> learn), genuine depth in the curated knowledge, and low-friction repeatability for non-technical users who won't write careful prompts themselves.

## Scope decision: three curated categories

Hand-authoring deep, compliance-aware knowledge for every possible D2C category doesn't scale. Settled on three representative examples instead of trying to generalize infinitely:

- **Kids nutrition** (SproutMix) - fictional, modeled on a real category's compliance structure
- **Fitness apparel** (FlexWear) - fictional
- **Skincare & personal care** (GlowLabs) - fictional, expanded to include a hair/body/supplement-gummies range

Each has brand voice + category-appropriate compliance rules (fixed, authoritative, loaded deterministically into every request - see "Compliance rules" in `README.md`) and hook banks/benchmarks (illustrative, not fixed, retrieved and ranked like any other knowledge - see the dynamic-vs-fixed framing in `core/reasoning/prompts.py`). Anything outside these three falls back to general D2C knowledge only, unless the user adds their own brand (see "Custom brands" below and in `README.md`) - the brand-picker UI exists specifically so users know which categories are actually well-supported instead of typing anything and getting ungrounded output.

**Custom brands aren't hardcoded.** The 3 curated brands were never meant to be the ceiling - "Add your brand" on the picker extracts a `BrandProfile` (voice & tone, positioning, personas, compliance rules) from an uploaded guide via one LLM call, lets the user review/edit it (compliance rules specifically require explicit confirmation before saving, since they become non-negotiable), and saves it to `data/brands/<slug>/`. From there it's a first-class brand - same `BrandProfile` shape as a preset, same retrieval scoping, same deterministic compliance injection, loaded alongside the 3 presets on the picker.

## Completed

- V0 scaffold: Python + Streamlit + ChromaDB, ingestion (CSV/XLSX/PDF), separate Explore/Connect collections, a pandas-based structured-data store, a heuristic query router, live Gemini integration via the current `google-genai` SDK
- Chat-based UI (replaced the original form-first flow) with a 4-card brand picker (3 curated brands + "Other brand") as the landing gate, example-prompt cards, idea cards with Brand data/Inference badges, follow-up chips, and a compact sidebar (data-in-use list with per-file remove, reset-to-sample-data, past chats, saved ideas)
- Sample data for all three brands, including a 100+ row fictional creative-testing dataset (ad format / language / narrative-angle combinations) for GlowLabs
- Fixed: Explore and Connect were mutually exclusive - loading any data (even from an unrelated category) silently blocked all curated Explore knowledge. Now merged, with a relevance-floor cutoff on `VectorStore.query()` so irrelevant chunks get dropped instead of forced into every prompt
- Fixed: even after that merge, a growing Explore knowledge base could still crowd out genuinely relevant Connect data (tabular sample-data chunks tend to score slightly worse than well-written prose). `core/reasoning/engine.py::_merge_chunks()` now reserves a minimum number of retrieval slots for Connect data whenever it's loaded and has anything relevant
- Fixed a real cross-session/cross-brand data leak: Connect data used to live in one shared collection, so one session's upload (or a lingering prior brand's data) could show up in another session's - or another brand's - answer. Connect is now one `connect_<uuid>` collection per browser session, cleared and reloaded on every brand switch; Explore chunks are tagged with a `brand` metadata field (from filename prefix) and every query is scoped to `[selected_brand, "general"]`
- Compliance/claim rules moved out of retrieval entirely: each brand's `<brand>_compliance.md` is excluded from the vector store and loaded deterministically off disk into the system prompt on every request (`core/knowledge/loader.py::load_compliance_rules`), under a `## Non-negotiable rules` heading - they no longer depend on vector search happening to rank them in the top-`k`. "Other brand" gets a short hardcoded general rule set instead
- `core/knowledge/loader.py::load_explore_knowledge` now rebuilds the Explore collection only when the knowledge files have actually changed (a content hash stored as Chroma collection metadata), instead of either always skipping a populated collection (silently stale after an edit) or always rebuilding (wasted embedding cost on every restart)
- Free-tier-only LLM reliability: replaced a single Gemini model + one hand-rolled retry loop with an ordered model chain (`core/reasoning/llm_client.py::ModelChainClient`) that walks configured models/providers in order - Gemini models (`GEMINI_MODELS`), then Groq (`GROQ_API_KEY`), then a local Ollama server (`OLLAMA_MODEL`) - advancing past any that's quota-exhausted, overloaded, or times out, raising a detailed `ChainExhaustedError` only if every entry fails. Structured JSON output via Gemini's `response_schema`, and a dev-only disk cache (`LLM_CACHE=1`) keyed by prompt hash
- Multi-turn follow-ups ("make idea 2 funnier", "Hindi versions of these") now actually carry context: `CopilotSession` keeps chat history and `ask()` passes the previous turn's ideas into the prompt and (for short/bare follow-ups) the previous query into retrieval; Refine/the "funnier" chip embed the specific idea's full content, not just its concept name
- Refactored so `app.py` is UI-only: `core/session.py::CopilotSession` owns all session state (brand, per-session Connect collection, chat history, saved ideas) and orchestrates requests (`ask()`); `ui/theme.py` and `ui/components.py` hold the CSS and render functions. `core/` has zero Streamlit dependency, enforced by a subprocess-isolated test
- An eval harness (`evals/`) - fixed per-brand prompts run through `CopilotSession`, checked for brand leakage, banned-claim phrases, and correct `needs_review` flagging, outside of pytest since it spends real API quota. Quota/network failures are recorded as a separate "error" status excluded from the pass rate rather than counted as failures; conditional (not absolutely-prohibited) banned phrases used with `needs_review` correctly set are warnings, not silently skipped; an optional `--judge` flag adds one semantic-compliance LLM call per idea, off by default
- "Add your brand": brands are no longer hardcoded to the 3 presets. `core/custom_brands.py` persists marketer-created brands (`data/brands/<slug>/`, gitignored) in the same `BrandProfile` shape as the presets (`core/brand_presets.py`); `core/reasoning/brand_extraction.py` makes one structured-output LLM call to extract voice/positioning/personas/compliance from an uploaded guide, reviewed and edited (compliance rules require explicit confirmation) before saving; a custom brand's guide is chunked into the shared Explore collection tagged `brand=<slug>` exactly like a preset's markdown, and its compliance is read deterministically from the saved profile instead of a static file - one code path either way
- "Export edit brief" on video-format ideas (Reels/UGC/video): `core/reasoning/edit_brief.py` makes one structured-output LLM call to turn an already-generated idea into a scene-by-scene `EditBrief` (never regenerating the idea itself), validated so scenes are contiguous and exactly span the stated duration, with compliance notes carried through from the idea's own `needs_review`/`review_reason`. Downloadable as JSON (the handoff format to external AI video tools - documented in `README.md`) or Markdown (human-readable)
- Ready for free deployment on Streamlit Community Cloud (see README's "Deploying" section): secrets read via `st.secrets` and copied into `os.environ` so `core/` stays Streamlit-free either way; a `pysqlite3-binary`/`sys.modules` swap for Chroma's SQLite requirement (Linux-only, marker-scoped so local `pip install` isn't affected); the Explore knowledge base already rebuilds automatically on the ephemeral disk's every reset; a custom brand can be downloaded/re-uploaded as a profile JSON to survive one; `OllamaClient` is skipped unless a local server actually responds; and quota protection for a public link on one shared key - an optional `DEMO_PASSWORD` gate, a per-session cap (`MAX_GENERATIONS_PER_SESSION`) on real generations, and `LLM_CACHE` forced off unless set in a genuine local `.env`

## In progress

(nothing currently in progress)

## Deferred / future ideas

- Deepening the knowledge base further with more real frameworks, as they become available (fictionalized, per the confidentiality approach established for SproutMix/FlexWear/GlowLabs)
- GTM mode for pre-launch companies with no product history - needs a decision on whether it's a third top-level mode or a branch of Explore
- On-demand synthetic data generation (LLM-generated, category-specific) as a follow-up to the static sample library
- Rate limiting / cost controls if this ever moves beyond local personal use
- Deployment (currently local-only, no hosting)
