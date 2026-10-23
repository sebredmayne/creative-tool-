"""The app's visual theme - one CSS block, injected once at startup. Kept separate from
app.py so styling changes never touch the widget/state wiring, and vice versa.

Palette/type/shape come from the "OnBrief redesign" mockup (ink on cream, Instrument Sans +
JetBrains Mono, a single yellow accent used for highlights and source chips, amber for
needs-review) - this file translates that mockup's hex values and type scale onto Streamlit's
fixed DOM via its data-testid hooks, rather than reproducing the mockup's custom HTML/layout
verbatim (Streamlit doesn't give that level of control over structure, only over the CSS
selectors below)."""

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #16161A;
    --text-secondary: #4A4A52;
    --text-body: #3A3A42;
    --bg-page: #FFFFFF;
    --bg-sidebar: #F5F5F2;
    --bg-tag: #EDECE7;
    --border: #E6E5E0;
    --border-strong: #CFCEC8;
    --accent: #FFE45C;
    --accent-soft: #FFFBE3;
    --accent-soft-border: #EFE2A0;
    --review-bg: #FDF1E3;
    --review-text: #8A4B0F;
    --review-border: #F0D9B8;
}

html, body, .stApp, p, span, label, li, div {
    font-family: 'Instrument Sans', system-ui, sans-serif;
    color: var(--ink);
}

h1, h2, h3, h4, .card-title, .idea-title {
    font-family: 'Instrument Sans', system-ui, sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em;
    color: var(--ink) !important;
}

.stApp {
    background-color: var(--bg-page);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--bg-page);
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: none;
}

/* Primary buttons: solid ink fill - reserved for the one standout action per screen */
[data-testid="stBaseButton-primary"] {
    background-color: var(--ink);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
[data-testid="stBaseButton-primary"]:hover {
    background-color: #2A2A30;
}
[data-testid="stBaseButton-primary"] p {
    color: inherit;
}

/* Default (secondary) buttons: outlined - Refine/Save/Copy/Change/Reset etc. */
[data-testid="stBaseButton-secondary"] {
    background-color: var(--bg-page);
    color: var(--ink);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    font-weight: 600;
}
[data-testid="stBaseButton-secondary"]:hover {
    background-color: var(--bg-sidebar);
    border-color: var(--ink);
}
[data-testid="stBaseButton-secondary"] p {
    color: inherit;
}

/* Tertiary buttons: flat text link, no border/fill - "Start chat ->", "Use this" */
[data-testid="stBaseButton-tertiary"] {
    background-color: transparent;
    color: var(--ink);
    border: none;
    font-weight: 600;
    padding-left: 0;
    justify-content: flex-start;
}
[data-testid="stBaseButton-tertiary"]:hover {
    color: var(--text-secondary);
    text-decoration: underline;
}
[data-testid="stBaseButton-tertiary"] p {
    color: inherit;
}

.card-title {
    font-size: 1.3rem;
    margin-bottom: 2px;
}

.idea-title {
    font-size: 1.15rem;
    margin: 0;
    font-weight: 700 !important;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
    border: 1px solid var(--border);
}

[data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea {
    border-radius: 10px !important;
}
[data-testid="stChatInput"] {
    border: 1.5px solid var(--ink) !important;
    border-radius: 12px !important;
}

/* Mono labels - section headers, timestamps, source filenames: matches the mockup's heavy use
   of JetBrains Mono for anything that reads as metadata rather than prose. */
.mono-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-secondary);
}

.idea-badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}
.badge-brand-data {
    background-color: var(--bg-tag);
    color: var(--ink);
}
.badge-inference {
    background-color: var(--accent-soft);
    color: var(--review-text);
}
.badge-custom {
    background-color: var(--accent);
    color: var(--ink);
}

/* Inline highlight behind a word/phrase in a heading - the mockup's treatment for a brand
   name or key phrase, e.g. "What do you want to make for <span class="highlight">X</span>?" */
.highlight {
    background-color: var(--accent);
    padding: 0 6px;
    border-radius: 2px;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}

/* "Drew from" source chip on an idea card - the one spot color shows up on a result, so what
   an idea actually drew from reads as distinct from prose, same as the mockup's source tags. */
.source-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 9px;
    border-radius: 4px;
    background-color: var(--accent);
    color: var(--ink);
}

.compliance-warning {
    background-color: var(--review-bg);
    border: 1px solid var(--review-border);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: var(--review-text);
}

</style>
"""
