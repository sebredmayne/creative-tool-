"""The app's visual theme - one CSS block, injected once at startup. Kept separate from
app.py so styling changes never touch the widget/state wiring, and vice versa."""

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, p, span, label, li, div {
    font-family: 'Inter', sans-serif;
    color: #1A1512;
}

h1, h2, h3, h4, .card-title, .idea-title {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    color: #1A1512 !important;
}

.stApp {
    background-color: #FAF7F2;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E7DFD0 !important;
    border-radius: 14px !important;
    box-shadow: none;
}

/* Default (secondary) buttons: outlined pill - Refine/Save/Copy/Change/Reset etc. */
[data-testid="stBaseButton-secondary"] {
    background-color: #FFFFFF;
    color: #C1502E;
    border: 1px solid #C1502E;
    border-radius: 999px;
    font-weight: 600;
}
[data-testid="stBaseButton-secondary"]:hover {
    background-color: #FBF0EA;
    color: #A5401F;
    border-color: #A5401F;
}
[data-testid="stBaseButton-secondary"] p {
    color: inherit;
}

/* Tertiary buttons: flat text link, no border/fill - "Start chat ->", "Use this" */
[data-testid="stBaseButton-tertiary"] {
    background-color: transparent;
    color: #C1502E;
    border: none;
    font-weight: 700;
    padding-left: 0;
    justify-content: flex-start;
}
[data-testid="stBaseButton-tertiary"]:hover {
    color: #A5401F;
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
}

.idea-badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    white-space: nowrap;
}
.badge-brand-data {
    background-color: #E3F0E3;
    color: #2F7D4F;
}
.badge-inference {
    background-color: #FBF0DC;
    color: #96700F;
}

.compliance-warning {
    background-color: #FBF0DC;
    border: 1px solid #E9D8A6;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #6B5410;
}

</style>
"""
