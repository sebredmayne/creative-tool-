# D2C Growth Copilot — V0

A creative-strategy tool with two ways to give the AI context:

- **Explore D2C** — tell it your brand/product/customer/objective, get ideas grounded in a curated, general D2C knowledge base.
- **Connect Data** — upload your own company data (CSV/XLSX/PDF), get ideas grounded in that data instead.

Both modes go through the same reasoning engine; they differ in which knowledge source it's allowed to draw from.

## Architecture

```
app.py                  Streamlit widgets/rendering only - one CopilotSession per browser
                        session lives in st.session_state; app.py just calls its methods
                        and renders whatever comes back
ui/
  theme.py              The CSS theme, injected once
  components.py         render_* functions (idea cards, follow-up chips) - take a
                        CopilotSession explicitly rather than reaching into st.session_state
core/
  session.py             CopilotSession - owns one session's brand selection, per-session
                          Connect collection, chat history, and request orchestration (ask())
  models.py               Shared dataclasses (ParsedDocument, QueryContext, CreativeIdea, ...)
  brand_presets.py         The 3 curated brands' sample data + starting context
  ingestion/               File -> ParsedDocument (text chunks + optional DataFrames)
  knowledge/explore/       Curated general + per-brand D2C knowledge (markdown), plus each
                           brand's *_compliance.md (excluded from retrieval - see below)
  knowledge/loader.py      Loads Explore knowledge into the vector store (hash-gated rebuild)
                           and loads a brand's compliance rules deterministically
  retrieval/               VectorStore (semantic search) + StructuredStore (pandas analysis)
  reasoning/               router -> prompts -> llm_client (free-tier model chain) -> engine
```

`core/` has no dependency on Streamlit - not just as a convention but enforced by a test
(`tests/test_session.py::test_core_session_never_imports_streamlit`, which imports
`core.session` in a subprocess and asserts `streamlit` never lands in `sys.modules`). It can be
tested and reused independently of the UI, so if the UI is ever rebuilt, none of this logic has
to move.

## How ingestion works

Every file type has a `Parser` (`core/ingestion/base.py`) that turns a file into a `ParsedDocument`:

- **CSV/XLSX** → a `pandas.DataFrame` per sheet (for structured analysis) *and* text chunks (a summary + small row groups written as `column: value` text, for semantic search).
- **PDF** → text chunks only; no structured data.

`core/ingestion/__init__.py` dispatches to the right parser by file extension. Adding DOCX/TXT later means writing one new `Parser` subclass and adding one line to that dispatch table — nothing else in the app changes.

## How retrieval works

Two separate pieces, because a numeric question and a "give me ideas" question need different tools:

1. **`VectorStore`** (`core/retrieval/vector_store.py`) — a local ChromaDB instance (no server, no API key — it uses a bundled local embedding model). It holds:
   - `explore_knowledge` — the curated markdown files in `core/knowledge/explore/`, one shared collection across all sessions. `core/knowledge/loader.py::load_explore_knowledge()` only rebuilds it when the knowledge files have actually changed: a hash of every source file's content is stored as Chroma collection metadata, so an unchanged knowledge base skips re-embedding on every process restart, while an edited/added `.md` file is picked up on the very next start.
   - one `connect_<uuid>` collection **per browser session** — chunks from whatever that session uploaded or loaded as sample data. Never shared between sessions (this used to be a single shared `connect_knowledge` collection, which caused real cross-brand/cross-user data leakage - one session's SproutMix upload showing up in another session's GlowLabs answer). `VectorStore.delete_collections_with_prefix("connect_")` sweeps up any left behind by a session that ended without cleanup (crash, restart, closed tab), once per process start.

   Explore queries are additionally scoped by brand: every Explore chunk is tagged with a `brand` metadata field (derived from its filename prefix - `sproutmix_...` → `"sproutmix"`, anything unprefixed → `"general"`), and `generate_ideas()` filters to `[selected_brand, "general"]` (or `["general"]` alone for "Other brand"), so a SproutMix question can never retrieve GlowLabs or FlexWear chunks.

   Explore and Connect are never queried against each other, so generic knowledge and a company's private data stay logically separate; `core/reasoning/engine.py::_merge_chunks()` merges their *results* afterward rather than querying one combined collection - and reserves a minimum number of slots for Connect data even when Explore's chunks score marginally better, since well-written prose otherwise tends to out-rank flatter tabular row-chunks even when the tabular data is the more relevant answer.

2. **`StructuredStore`** (`core/retrieval/structured_store.py`) — holds the raw DataFrames from uploaded CSV/XLSX in memory and produces plain-text summaries (shape, columns, sample rows, numeric stats) for questions that need actual computation rather than semantic similarity (e.g. "what's our average order value?"). V0 deliberately does *not* execute arbitrary generated code against the data — only these canned, safe summaries.

**Routing** (`core/reasoning/router.py`) decides, per query, whether to use semantic retrieval, structured summaries, or both. V0 uses keyword heuristics (e.g. "average", "how many", "%") rather than an LLM call, so it works without any API key and is simple to unit test. This is the seam where a smarter (LLM-based) router could be swapped in later.

## Explore vs. Connect

| | Explore | Connect |
|---|---|---|
| Knowledge source | Curated general + brand-scoped D2C knowledge (`explore_knowledge` collection, filtered by brand) | User's uploaded/sample files (that session's own `connect_<uuid>` collection) |
| Extra context | Brand/product/customer/category/objective form fields | Whatever's in the uploaded files |
| Structured data | Never available (a fresh, empty `StructuredStore` is used) | Available if CSV/XLSX was uploaded |

Both call the same `core/reasoning/engine.py::generate_ideas()` — only the collection name and the available context/structured data differ. This is enforced in code, not just by convention: Explore always passes a brand-new empty `StructuredStore`, so nothing uploaded in a Connect session can leak into an Explore answer even within the same browser session; whether a query runs in Explore or Connect mode is decided per-request in `CopilotSession.ask()` based on whether any Connect data is currently loaded, not a separate mode the user has to pick.

## Compliance rules

Brand voice, hooks, and benchmarks are retrieved like any other knowledge - useful, but not
guaranteed to rank in the top-`k` for a given query. Compliance/claim rules can't work that way:
a rule that might silently drop out of the prompt isn't a rule. So each brand's compliance
content lives in its own `<brand>_compliance.md` file (e.g. `sproutmix_compliance.md`),
excluded from the vector store entirely (`core/knowledge/loader.py` skips any `*_compliance.md`
when building the Explore collection) and instead loaded straight off disk, deterministically,
by `load_compliance_rules()` on every single request - regardless of what the query is or what
retrieval returns. `build_system_prompt()` injects the result under an explicit `## Non-negotiable rules`
heading, with wording that tells the model these override anything retrieved or asked for. A
brand-less session ("Other brand") gets a short hardcoded fallback (`GENERAL_COMPLIANCE_RULES`
in `core/knowledge/loader.py`): no medical/efficacy claims, flag anything health-related for review.

## Multi-turn follow-ups

A follow-up like "make idea 2 funnier" or "Hindi versions of these" is meaningless in isolation
- there's no "idea 2" or "these" in a single-shot prompt. `CopilotSession` keeps chat history
(`self.messages`), and `ask()` pulls the previous turn's ideas and query out of it before
generating:

- The previous turn's ideas (concept, format, and a truncated script) are always included in
  the prompt under an "Ideas from the previous turn" heading - `build_user_prompt()`'s
  `previous_ideas` parameter.
- For retrieval specifically, a short or clearly-a-follow-up query (under ~8 words, or starting
  with "make"/"now", or a "give me ... versions" shape) also has the *previous* query appended
  to the text actually sent to `VectorStore.query()` - see `_needs_previous_query_context()` in
  `core/reasoning/engine.py` - so a bare "make it funnier" doesn't retrieve nothing just because
  it has no topical content of its own. The prompt's visible "User request" line still shows
  only the new query; only the retrieval text is augmented.
- The Refine button and the "funnier" follow-up chip additionally embed the specific idea's
  full content (not just its concept name) directly into the follow-up query, so that specific
  idea is unambiguous even before the previous-turn context is considered.

## LLM provider

Isolated entirely behind `core/reasoning/llm_client.py`. This project intentionally never uses a paid key - reliability comes from trying several free-tier models/providers in order, not from paying for one that doesn't fall over.

`get_llm_client()` builds an ordered **model chain** from whatever's configured in the environment, and `ModelChainClient` walks it on every request: a model that's quota-exhausted (`429`), overloaded (`503`), times out, or otherwise rejects the request is skipped in favor of the next one, stopping at the first success. Only if *every* entry fails does it raise `ChainExhaustedError`, which lists what each model actually returned (so a failure is debuggable, not a generic "try again").

- Nothing configured at all → falls back to `MockLLMClient`, which returns a fixed, validly-shaped response so the whole app runs end-to-end without any credentials.
- `GEMINI_API_KEY` set → adds one chain entry per model in `GEMINI_MODELS` (comma-separated; the `google-genai` package is imported lazily, only when this path is actually used). Each entry uses Gemini's `system_instruction` (rather than concatenating the system prompt into the user message), `response_mime_type="application/json"` with a `response_schema` matching `CreativeIdea`'s fields (so `_parse_ideas`'s manual JSON parsing is a safety net that should rarely trigger, not the primary path), and `thinking_config.thinking_budget=0` for speed.
- `GROQ_API_KEY` set → appends a `GroqClient` entry, talking to Groq's OpenAI-compatible endpoint directly over HTTP - a second, independent free provider so a request can still succeed even if every Gemini model is unavailable at once.
- `OLLAMA_MODEL` set → appends an `OllamaClient` entry, talking to a local Ollama server - the final fallback, with zero API key and zero network dependency.
- `LLM_CACHE=1` → wraps the chain in a dev-only disk cache (`data/llm_cache/`, gitignored) keyed by a hash of `(system_prompt, user_prompt)`. A cache hit skips every provider entirely. Never enable this for an actual demo - it would silently mask whether real inference is happening for a given input.

To connect real keys: copy `.env.example` to `.env` and fill in whichever of `GEMINI_API_KEY` / `GROQ_API_KEY` / `OLLAMA_MODEL` you have. Nothing else needs to change. Note: Google's older `google-generativeai` package is fully deprecated as of late 2026 — this project uses the current `google-genai` SDK instead.

## Output shape

Every generated idea (`core/models.py::CreativeIdea`) has: `concept`, `rationale`, `recommended_format`, `source_context` (what knowledge/data it drew from), and an optional `script`.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

No `.env` file is required to run V0 — it works out of the box using the mock LLM client.

## Running tests

```bash
pytest
```

Tests use only synthetic, made-up data (no real company/customer data), and don't require any API key.

## Evals

`evals/` is a separate, fixed-prompt eval harness - deliberately **not** part of `pytest`, since
it makes real LLM calls (spending real free-tier quota) rather than using mocks:

```bash
python -m evals.run           # mechanical checks only
python -m evals.run --judge   # + one extra LLM call per idea, semantic compliance review
```

`evals/prompts.yaml` has 5-8 fixed prompts per curated brand (plus each brand's list of banned
claim phrases, drawn from its `<brand>_compliance.md`), run through the real `CopilotSession`
path. Every prompt ends in one of three states, not two:

- **error** - every configured model was quota-exhausted, overloaded, or timed out, so nothing
  even generated. That's a statement about today's free-tier capacity, not about this app, so
  it's reported separately and **excluded from the pass rate** rather than counted as a fail.
- **fail** - it generated, but a check found a real problem: invalid structure (the
  `_parse_ideas` fallback), a mention of another brand, a banned claim phrase used without being
  flagged, an expected `needs_review` that never fired, an implied-claim match (see below), or
  (with `--judge`) a semantic compliance finding.
- **pass** - everything else.

Banned-phrase checks aren't all treated the same way: SproutMix's compliance file lists phrases
that are "no exceptions" - presence alone is always a fail. FlexWear/GlowLabs's phrases are
conditional on substantiation the brand doesn't have, and their own compliance file says that
should get a soft warning, not an automatic hard stop - so presence there is a **warning**
(printed, not silently skipped) if the idea correctly flagged it for review, and only a fail if
it didn't (`banned_phrases_are_absolute` in `prompts.yaml` controls which rule applies).

One SproutMix case checks something banned phrases can't catch at all: `forbidden_implied_claims`
fails a prompt if its output implies a mood/behaviour/growth/immunity/brain outcome via any of a
small keyword list, even without an exact banned phrase - SproutMix's real compliance risk is as
much about implied outcomes ("supports healthy development") as literal claims. `--judge` is the
semantic backstop for whatever that keyword list misses: one extra LLM call per idea, asking the
model itself whether the idea complies with that brand's actual compliance rules. Off by default
since it roughly doubles the API calls a run makes.

Prints a table and saves the full results to `evals/results/<timestamp>.json` (gitignored).
Sets `LLM_CACHE=1` by default so repeated runs during development reuse prior responses instead
of spending quota every time - overriding whatever `.env` says, since evals specifically want
this even when the app's own default is cache-off.

## Out of scope for V0

- Cardboard integration
- Authentication, payments, deployment infrastructure
- Persistent accounts or a shared database - each browser session is correctly isolated (its
  own Connect data, chat history) while the server is running, but nothing survives past
  process restart except the shared Explore knowledge base itself
- LLM-based query routing (currently keyword heuristics)
- Arbitrary code generation/execution for structured data analysis (only canned, safe summaries)
- DOCX/TXT ingestion (interface supports adding it; not implemented yet)
