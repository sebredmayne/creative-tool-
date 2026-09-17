# D2C Growth Copilot — V0

A creative-strategy tool with two ways to give the AI context:

- **Explore D2C** — tell it your brand/product/customer/objective, get ideas grounded in a curated, general D2C knowledge base.
- **Connect Data** — upload your own company data (CSV/XLSX/PDF), get ideas grounded in that data instead.

Both modes go through the same reasoning engine; they differ in which knowledge source it's allowed to draw from.

## Architecture

```
app.py                  Streamlit UI only — no business logic
core/
  models.py             Shared dataclasses (ParsedDocument, QueryContext, CreativeIdea, ...)
  ingestion/             File -> ParsedDocument (text chunks + optional DataFrames)
  knowledge/explore/     Curated general D2C knowledge (markdown)
  retrieval/             VectorStore (semantic search) + StructuredStore (pandas analysis)
  reasoning/             router -> prompts -> llm_client -> engine (orchestration)
```

`core/` has no dependency on Streamlit. It can be tested and reused independently of the UI, so if the UI is ever rebuilt, none of this logic has to move.

## How ingestion works

Every file type has a `Parser` (`core/ingestion/base.py`) that turns a file into a `ParsedDocument`:

- **CSV/XLSX** → a `pandas.DataFrame` per sheet (for structured analysis) *and* text chunks (a summary + small row groups written as `column: value` text, for semantic search).
- **PDF** → text chunks only; no structured data.

`core/ingestion/__init__.py` dispatches to the right parser by file extension. Adding DOCX/TXT later means writing one new `Parser` subclass and adding one line to that dispatch table — nothing else in the app changes.

## How retrieval works

Two separate pieces, because a numeric question and a "give me ideas" question need different tools:

1. **`VectorStore`** (`core/retrieval/vector_store.py`) — a local ChromaDB instance (no server, no API key — it uses a bundled local embedding model). It holds **two separate collections**:
   - `explore_knowledge` — the curated markdown files in `core/knowledge/explore/`, loaded once at startup.
   - `connect_knowledge` — chunks from whatever the user uploads in Connect mode.
   These are never queried against each other, so Explore's generic knowledge and a company's private data stay logically separate.

2. **`StructuredStore`** (`core/retrieval/structured_store.py`) — holds the raw DataFrames from uploaded CSV/XLSX in memory and produces plain-text summaries (shape, columns, sample rows, numeric stats) for questions that need actual computation rather than semantic similarity (e.g. "what's our average order value?"). V0 deliberately does *not* execute arbitrary generated code against the data — only these canned, safe summaries.

**Routing** (`core/reasoning/router.py`) decides, per query, whether to use semantic retrieval, structured summaries, or both. V0 uses keyword heuristics (e.g. "average", "how many", "%") rather than an LLM call, so it works without any API key and is simple to unit test. This is the seam where a smarter (LLM-based) router could be swapped in later.

## Explore vs. Connect

| | Explore | Connect |
|---|---|---|
| Knowledge source | Curated general D2C knowledge (`explore_knowledge` collection) | User's uploaded files (`connect_knowledge` collection) |
| Extra context | Brand/product/customer/category/objective form fields | Whatever's in the uploaded files |
| Structured data | Never available (a fresh, empty `StructuredStore` is used) | Available if CSV/XLSX was uploaded |

Both call the same `core/reasoning/engine.py::generate_ideas()` — only the collection name and the available context/structured data differ. This is enforced in `app.py`, not just by convention: Explore always passes a brand-new empty `StructuredStore`, so nothing uploaded in a Connect session can leak into an Explore answer even within the same browser session.

## LLM provider

Isolated entirely behind `core/reasoning/llm_client.py`. `get_llm_client()` reads `LLM_PROVIDER`, `GEMINI_API_KEY`, and `GEMINI_MODEL` from the environment:

- No `GEMINI_API_KEY` set → falls back to `MockLLMClient`, which returns a fixed, validly-shaped response so the whole app runs end-to-end without any credentials.
- `GEMINI_API_KEY` set → uses `GeminiClient` (the `google-generativeai` package is imported lazily, only when this path is actually used).

To connect a real key later: copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`. Nothing else needs to change. (The Gemini call in `GeminiClient.generate()` hasn't been exercised against the live API yet — worth a quick sanity check against the current `google-generativeai` SDK once a key is added.)

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

## Out of scope for V0

- Cardboard integration
- Authentication, payments, deployment infrastructure
- Multi-user support or a shared database (data is in-memory / local-disk, single user)
- LLM-based query routing (currently keyword heuristics)
- Arbitrary code generation/execution for structured data analysis (only canned, safe summaries)
- DOCX/TXT ingestion (interface supports adding it; not implemented yet)
