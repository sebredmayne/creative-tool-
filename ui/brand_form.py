"""The "Add your brand" flow: a form, one LLM extraction call, then an editable review screen
before anything is saved. Two steps driven by st.session_state.brand_creation_step so the
in-progress draft survives Streamlit reruns between them.

Compliance rules specifically require an explicit checkbox confirmation before the "Save
brand" button is enabled - once saved, they're injected as non-negotiable into every request
for this brand (core.knowledge.loader.load_compliance_rules), so silently trusting an
unreviewed extraction there would be the one mistake most worth avoiding.
"""
import json
from dataclasses import asdict

import streamlit as st

from core.custom_brands import get_brand, save_custom_brand, slugify
from core.ingestion import get_parser
from core.knowledge.loader import load_explore_knowledge
from core.models import BrandProfile
from core.reasoning.brand_extraction import BrandExtractionError, extract_brand_profile_fields

_PROFILE_RESTORE_FIELDS = ("brand_name", "voice_and_tone", "positioning", "personas", "compliance_rules")


def _read_guide_text(file) -> str:
    parser = get_parser(file.name)
    if parser is None:
        raise ValueError(f"Unsupported guide file type: {file.name}. Use PDF, MD, or TXT.")
    parsed = parser.parse(file, file.name)
    return "\n\n".join(parsed.text_chunks)


def _exit_creation_flow() -> None:
    st.session_state.brand_creation_step = None
    st.session_state.pop("brand_creation_draft", None)


def parse_restored_profile_json(raw_bytes: bytes) -> dict:
    """Parses a previously-downloaded brand profile JSON into a brand_creation_draft dict -
    the pure logic half of the restore flow, kept separate from the widgets around it so it's
    testable without a Streamlit runtime. Raises ValueError (with a message fit to show the
    user directly) if the file isn't a valid profile.

    Only restores the profile's own fields (voice/positioning/personas/compliance/context) -
    not the guide's full illustrative text or any data files, which weren't part of the
    download; the caller resolves slug collisions itself since that needs get_brand()."""
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
        brand_name = str(data["brand_name"])
        for field_name in _PROFILE_RESTORE_FIELDS:
            if field_name not in data:
                raise KeyError(field_name)
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError, TypeError) as e:
        raise ValueError(f"That doesn't look like a valid brand profile file: {e}") from e

    context = data.get("context") or {}
    return {
        "slug": data.get("slug") or slugify(brand_name),
        "brand_name": brand_name,
        "category": data.get("label") or context.get("category", ""),
        "target_customer": context.get("customer", ""),
        "objective": context.get("objective", ""),
        "guide_filename": data.get("guide_filename") or "restored_profile.json",
        "guide_bytes": b"",
        "guide_text": "",
        "data_files": [],
        "voice_and_tone": data["voice_and_tone"],
        "positioning": data["positioning"],
        "personas": data["personas"],
        "compliance_rules": data["compliance_rules"],
    }


def _render_profile_restore() -> None:
    """Restores a brand from a previously downloaded profile JSON (see the "Download brand
    profile" button on the review screen) - skips extraction entirely and jumps straight to
    the same review/confirm/save step a fresh extraction would, so a restored brand's
    compliance rules still require the same explicit re-confirmation before saving."""
    st.caption(
        "Restores voice/positioning/personas/compliance and basic context. Re-upload any "
        "reviews/sales data files separately afterward if you want those back too."
    )
    restore_file = st.file_uploader("Brand profile JSON", type=["json"], key="restore_profile_upload")
    if restore_file is None:
        return
    if not st.button("Restore this profile →", key="restore_profile_button"):
        return

    try:
        draft = parse_restored_profile_json(restore_file.getvalue())
    except ValueError as e:
        st.error(str(e))
        return

    if get_brand(draft["slug"]) is not None:
        draft["slug"] = slugify(draft["brand_name"])  # slug taken (e.g. by a preset) - pick a fresh one

    st.session_state.brand_creation_draft = draft
    st.session_state.brand_creation_step = "review"
    st.rerun()


def render_creation_flow(vector_store, llm_client) -> None:
    """Renders whichever step of brand creation is currently active."""
    step = st.session_state.get("brand_creation_step")
    if step == "review":
        _render_review(vector_store)
    else:
        _render_form(llm_client)


def _render_form(llm_client) -> None:
    st.markdown("### Add your brand")
    st.caption(
        "Upload a brand guide and we'll extract voice & tone, positioning, personas, and "
        "compliance rules for you to review before anything is saved."
    )
    st.info(
        "On a hosted/ephemeral deployment (e.g. Streamlit Community Cloud), the disk resets on "
        "every restart - custom brands saved there are temporary. Download a brand profile "
        "(JSON) after saving to back it up, and restore it below if it's ever lost.",
        icon="💾",
    )
    if st.button("← Back", key="brand_form_back"):
        _exit_creation_flow()
        st.rerun()

    with st.expander("Restore from a previously downloaded brand profile"):
        _render_profile_restore()

    with st.form("brand_creation_form"):
        brand_name = st.text_input("Brand name")
        category = st.text_input("Category", placeholder="e.g. Skincare, Kids nutrition, Fitness apparel")
        target_customer = st.text_input("Target customer")
        objective = st.text_input("Business objective", placeholder="e.g. Acquisition, retention")
        guide_file = st.file_uploader("Brand guide (PDF, MD, or TXT)", type=["pdf", "md", "txt"])
        data_files = st.file_uploader(
            "Optional: reviews/sales data (CSV/XLSX)", type=["csv", "xlsx"], accept_multiple_files=True
        )
        submitted = st.form_submit_button("Extract brand profile →")

    if not submitted:
        return

    if not brand_name.strip():
        st.error("Brand name is required.")
        return
    if guide_file is None:
        st.error("A brand guide file is required.")
        return

    with st.spinner("Reading your brand guide and extracting a profile..."):
        try:
            guide_text = _read_guide_text(guide_file)
            extracted = extract_brand_profile_fields(guide_text, llm_client)
        except BrandExtractionError as e:
            st.error(f"Couldn't extract a brand profile from that guide: {e}")
            return
        except Exception as e:
            st.error(f"Something went wrong reading that file: {e}")
            return

    st.session_state.brand_creation_draft = {
        "slug": slugify(brand_name),
        "brand_name": brand_name.strip(),
        "category": category.strip(),
        "target_customer": target_customer.strip(),
        "objective": objective.strip(),
        "guide_filename": guide_file.name,
        "guide_bytes": guide_file.getvalue(),
        "guide_text": guide_text,
        "data_files": [(f.name, f.getvalue()) for f in (data_files or [])],
        **extracted,
    }
    st.session_state.brand_creation_step = "review"
    st.rerun()


def _render_review(vector_store) -> None:
    draft = st.session_state.get("brand_creation_draft")
    if draft is None:
        _exit_creation_flow()
        st.rerun()
        return

    st.markdown("### Review your brand profile")
    st.caption(
        f"Extracted from {draft['guide_filename']}. Edit anything before saving - the model's "
        "first pass isn't guaranteed to be perfect."
    )

    voice_and_tone = st.text_area("Voice & tone", value=draft["voice_and_tone"], height=150)
    positioning = st.text_area("Positioning", value=draft["positioning"], height=150)
    personas = st.text_area("Personas", value=draft["personas"], height=120)

    st.markdown("#### Compliance & claim rules")
    st.caption(
        "These become non-negotiable once saved: injected into every request for this brand, "
        "overriding anything else retrieved or asked for."
    )
    compliance_rules = st.text_area("Compliance rules", value=draft["compliance_rules"], height=200)
    confirmed = st.checkbox("I've reviewed these compliance rules and confirm they're accurate.")

    # Built from the live (possibly just-edited) field values, so "Download brand profile"
    # reflects what's on screen right now, whether or not it's been saved yet.
    profile = BrandProfile(
        slug=draft["slug"],
        label=draft["category"] or draft["brand_name"],
        brand_name=draft["brand_name"],
        description=positioning[:140] + ("..." if len(positioning) > 140 else ""),
        context={
            "brand": draft["brand_name"],
            "product": "",
            "customer": draft["target_customer"],
            "category": draft["category"],
            "objective": draft["objective"],
        },
        voice_and_tone=voice_and_tone,
        positioning=positioning,
        personas=personas,
        compliance_rules=compliance_rules,
        sample_files=[filename for filename, _ in draft["data_files"]],
        example_prompts=[],
        guide_filename=draft["guide_filename"],
        is_custom=True,
    )

    col_back, col_download, col_save = st.columns(3)
    with col_back:
        if st.button("← Start over", key="brand_review_back", use_container_width=True):
            _exit_creation_flow()
            st.rerun()
    with col_download:
        st.download_button(
            "Download brand profile (JSON)",
            data=json.dumps(asdict(profile), indent=2),
            file_name=f"{profile.slug}_profile.json",
            mime="application/json",
            key="brand_review_download",
            use_container_width=True,
        )
    with col_save:
        if st.button(
            "Save brand", key="brand_review_save", type="primary", use_container_width=True, disabled=not confirmed
        ):
            save_custom_brand(
                profile,
                guide_text=draft["guide_text"],
                guide_filename=draft["guide_filename"],
                guide_bytes=draft["guide_bytes"],
                data_files=draft["data_files"],
            )
            # The new brand's guide just changed the knowledge hash - rebuild now rather than
            # waiting for the next process restart, so it's usable immediately.
            load_explore_knowledge(vector_store)

            st.session_state.new_brand_slug = profile.slug
            _exit_creation_flow()
            st.rerun()
