"""Eval harness: runs a fixed set of prompts per brand through the real CopilotSession/
generate_ideas path and checks for brand leakage, banned compliance-claim phrases, and correct
needs_review flagging.

Deliberately NOT part of pytest - it makes real LLM calls (spending real free-tier quota) unless
LLM_CACHE is warm. Run with:

    python -m evals.run           # mechanical checks only
    python -m evals.run --judge   # + one extra LLM call per idea, semantic compliance review

Always uses the dev response cache, overriding LLM_CACHE from .env if it's set to something
else - evals specifically want repeated runs during development to re-use prior responses
instead of burning quota every time (a plain `setdefault` isn't enough for this: .env sets
LLM_CACHE=0 for the app's own default, and load_dotenv() populates os.environ with that before
this module gets a chance to touch it, so a later setdefault would see the key already present
and do nothing).

Every prompt ends in one of three states, not two:
  - "pass"  - generated fine, every check was satisfied.
  - "fail"  - generated fine, but a check found a real problem (banned phrase used without
    being flagged, brand leakage, bad structure, a semantic judge finding, etc.).
  - "error" - didn't generate at all, because every configured model was quota-exhausted,
    overloaded, or timed out. That's a statement about today's free-tier capacity, not about
    whether the app or its compliance rules work - so it's reported separately and excluded
    from the pass rate rather than counted as a "fail".
"""
import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()
os.environ["LLM_CACHE"] = "1"

from core.brand_presets import BRAND_PRESETS
from core.knowledge.loader import load_compliance_rules, load_explore_knowledge
from core.models import CreativeIdea
from core.reasoning.engine import strip_code_fence
from core.reasoning.llm_client import LLMClient, get_llm_client
from core.retrieval.vector_store import VectorStore
from core.session import CopilotSession

EVALS_DIR = Path(__file__).parent
PROMPTS_FILE = EVALS_DIR / "prompts.yaml"
RESULTS_DIR = EVALS_DIR / "results"

# Keyword groups for the implied-outcome-claim check (see the sproutmix prompt tagged
# forbidden_implied_claims) - these catch language that implies one of these outcomes without
# necessarily using an exact banned phrase, e.g. "helps them settle down for bed" implies mood/
# behaviour even though it never says "calmer" or "boosts immunity" verbatim... this list is a
# heuristic starting point, not exhaustive; --judge is the semantic backstop for what it misses.
IMPLIED_CLAIM_KEYWORDS = {
    "mood": ["mood", "cranky", "crankiness", "tantrum", "happier", "calmer", "calm down"],
    "behaviour": ["behaviour", "behavior", "well-behaved", "well behaved", "listens better", "misbehav"],
    "growth": ["grow", "growth", "taller", "height", "development"],
    "immunity": ["immunity", "immune", "sick less", "fewer sick days", "fight off"],
    "brain": ["brain", "smarter", "focus", "concentration", "memory", "cognitive"],
}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict compliance reviewer for D2C marketing creative. You will be given one "
    "brand's non-negotiable compliance rules and one piece of already-generated creative "
    "(concept, format, rationale, script). Decide whether the creative violates any rule - "
    "including claims that are only implied, not just verbatim banned phrases. Respond with "
    'JSON only, no other text: {"compliant": true or false, "issues": ["short issue", ...]}. '
    "If compliant, issues must be an empty list."
)


@dataclass
class PromptResult:
    brand: str
    prompt: str
    status: str  # "pass" | "fail" | "error"
    num_ideas: int
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _other_brand_markers(brand_key: str) -> list[str]:
    """Distinctive name fragments for every brand *other* than `brand_key` - if any of these
    show up in an idea, that's a brand-leakage failure."""
    markers = []
    for key, preset in BRAND_PRESETS.items():
        if key == brand_key:
            continue
        markers.append(preset["brand_name"])
        markers.append(preset["brand_name"].split(" / ")[0])
        markers.append(preset["context"]["product"])
    return sorted({m for m in markers if m})


def _idea_text(idea: CreativeIdea) -> str:
    return " ".join(filter(None, [idea.concept, idea.rationale, idea.script or "", idea.source_context]))


def _structure_failures(idea: CreativeIdea) -> list[str]:
    if idea.concept == "Unparsed model output" or idea.recommended_format == "n/a":
        return ["invalid structure: unparsed model output"]

    failures = []
    for field_name in ("concept", "rationale", "recommended_format"):
        if not isinstance(getattr(idea, field_name), str) or not getattr(idea, field_name).strip():
            failures.append(f"missing/invalid {field_name}")
    if idea.grounding not in ("brand_data", "inference"):
        failures.append(f"invalid grounding value: {idea.grounding!r}")
    if not isinstance(idea.needs_review, bool):
        failures.append("needs_review is not a bool")
    return failures


def check_idea(
    idea: CreativeIdea, brand_key: str, banned_phrases: list[str], phrases_are_absolute: bool
) -> tuple[list[str], list[str]]:
    """Returns (failures, warnings). `phrases_are_absolute` matches how the brand's own
    compliance file treats these phrases:

    - True (SproutMix): its "Absolutely prohibited phrases" list is explicitly "no exceptions" -
      presence alone is always a failure, regardless of needs_review.
    - False (FlexWear, GlowLabs): these phrases are conditional on substantiation the brand
      doesn't have ("without an actual test/certification" / "not actually backed by ... data"),
      and their compliance file's own response protocol says that should get a Soft warning,
      not an automatic Hard stop - so presence is a *warning* (reported, not skipped) if the
      idea correctly flagged it for review, and only a failure if it didn't. (Found by running
      this for real: without this distinction, "true to size" in an ordinary FlexWear fit claim
      failed every time even when needs_review was correctly set, which isn't the actual rule.)
    """
    structure_failures = _structure_failures(idea)
    if structure_failures:
        return structure_failures, []  # nothing else is meaningful to check on unparsed output

    failures: list[str] = []
    warnings: list[str] = []
    text = _idea_text(idea).lower()

    for marker in _other_brand_markers(brand_key):
        if marker.lower() in text:
            failures.append(f"mentions another brand: {marker!r}")

    hit_phrases = [phrase for phrase in banned_phrases if phrase.lower() in text]
    if hit_phrases:
        if phrases_are_absolute:
            failures.append(f"uses banned claim phrase(s): {hit_phrases}")
            if not idea.needs_review:
                failures.append("banned phrase present but needs_review is False")
        elif idea.needs_review:
            warnings.append(f"uses conditional claim phrase(s), correctly flagged for review: {hit_phrases}")
        else:
            failures.append(f"uses conditional claim phrase(s) without needs_review: {hit_phrases}")

    return failures, warnings


def check_implied_claims(idea: CreativeIdea, categories: list[str]) -> list[str]:
    """Fails if the idea's text implies one of the given outcome categories (mood, behaviour,
    growth, immunity, brain) via any of its known keywords - catches implied claims that don't
    use an exact banned phrase."""
    text = _idea_text(idea).lower()
    failures = []
    for category in categories:
        for keyword in IMPLIED_CLAIM_KEYWORDS.get(category, []):
            if keyword in text:
                failures.append(f"implies a {category} claim (matched {keyword!r})")
    return failures


def judge_idea(idea: CreativeIdea, brand_key: str, llm_client: LLMClient) -> list[str]:
    """One extra LLM call: asks the model itself whether this idea complies with the brand's
    actual compliance rules, semantically - catches violations the mechanical phrase/keyword
    checks above can't, at the cost of an extra API call per idea. Only run with --judge."""
    compliance_rules = load_compliance_rules(brand_key)
    idea_description = (
        f'Concept: "{idea.concept}"\n'
        f"Format: {idea.recommended_format}\n"
        f"Rationale: {idea.rationale}\n"
        f"Script: {idea.script or '(none)'}"
    )
    user_prompt = f"Brand's non-negotiable compliance rules:\n{compliance_rules}\n\nCreative to review:\n{idea_description}"

    try:
        raw = llm_client.generate(JUDGE_SYSTEM_PROMPT, user_prompt)
        parsed = json.loads(strip_code_fence(raw))
    except Exception as e:
        return [f"LLM judge: could not get/parse a verdict ({type(e).__name__})"]

    if parsed.get("compliant", True):
        return []
    issues = parsed.get("issues") or ["flagged non-compliant, no specific issue given"]
    return [f"LLM judge: {issue}" for issue in issues]


def _run_one_prompt(
    vector_store, llm_client: LLMClient, brand_key: str, banned_phrases: list[str], phrases_are_absolute: bool,
    prompt_entry, use_judge: bool,
) -> PromptResult:
    if isinstance(prompt_entry, dict):
        prompt_text = prompt_entry["text"]
        expect_needs_review = prompt_entry.get("expect_needs_review", False)
        forbidden_implied_claims = prompt_entry.get("forbidden_implied_claims", [])
    else:
        prompt_text = prompt_entry
        expect_needs_review = False
        forbidden_implied_claims = []

    # A fresh session per prompt: each prompt is evaluated in isolation (not as a continuing
    # conversation), so results don't depend on prompt order within a brand's list.
    session = CopilotSession(vector_store, llm_client)
    session.load_brand(brand_key)
    result = session.ask(prompt_text, brand_context=BRAND_PRESETS[brand_key]["context"])

    if result.error:
        if result.error_is_provider_side:
            # Every model was quota-exhausted/overloaded/timed out - a statement about today's
            # free-tier capacity, not about this app, so it's kept out of the pass rate entirely.
            return PromptResult(brand=brand_key, prompt=prompt_text, status="error", num_ideas=0, failures=[result.error])
        return PromptResult(
            brand=brand_key, prompt=prompt_text, status="fail", num_ideas=0, failures=[f"generation error: {result.error}"]
        )

    failures: list[str] = []
    warnings: list[str] = []
    for idea in result.ideas:
        idea_failures, idea_warnings = check_idea(idea, brand_key, banned_phrases, phrases_are_absolute)
        failures.extend(idea_failures)
        warnings.extend(idea_warnings)
        if forbidden_implied_claims:
            failures.extend(check_implied_claims(idea, forbidden_implied_claims))
        if use_judge:
            failures.extend(judge_idea(idea, brand_key, llm_client))

    if expect_needs_review and not any(idea.needs_review for idea in result.ideas):
        failures.append("expected at least one idea flagged needs_review=True, but none were")

    return PromptResult(
        brand=brand_key,
        prompt=prompt_text,
        status="fail" if failures else "pass",
        num_ideas=len(result.ideas),
        failures=failures,
        warnings=warnings,
    )


def run(use_judge: bool = False) -> list[PromptResult]:
    brand_specs = yaml.safe_load(PROMPTS_FILE.read_text())

    vector_store = VectorStore()
    load_explore_knowledge(vector_store)
    llm_client = get_llm_client()

    results = []
    for brand_key, brand_spec in brand_specs.items():
        banned_phrases = brand_spec.get("banned_phrases", [])
        phrases_are_absolute = brand_spec.get("banned_phrases_are_absolute", False)
        for prompt_entry in brand_spec["prompts"]:
            results.append(
                _run_one_prompt(
                    vector_store, llm_client, brand_key, banned_phrases, phrases_are_absolute, prompt_entry, use_judge
                )
            )
    return results


def print_table(results: list[PromptResult]) -> None:
    print(f"{'BRAND':<12}{'STATUS':<8}PROMPT")
    print("-" * 90)
    for r in results:
        prompt_preview = r.prompt if len(r.prompt) <= 65 else r.prompt[:62] + "..."
        print(f"{r.brand:<12}{r.status.upper():<8}{prompt_preview}")
        for failure in r.failures:
            print(f"{'':<20}- {failure}")
        for warning in r.warnings:
            print(f"{'':<20}! {warning}")
    print("-" * 90)

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    errored = sum(1 for r in results if r.status == "error")
    scored = passed + failed
    rate = f"{passed}/{scored}" if scored else "n/a (everything errored)"
    print(f"{rate} passed", end="")
    print(f" - {errored} errored, excluded from pass rate" if errored else "")


def save_results(results: list[PromptResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{timestamp}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the D2C Growth Copilot eval suite.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Additionally run one LLM call per idea to check compliance semantically. Off by "
        "default - it roughly doubles+ the number of API calls this run makes.",
    )
    args = parser.parse_args()

    results = run(use_judge=args.judge)
    print_table(results)
    path = save_results(results)
    print(f"\nSaved results to {path}")
    sys.exit(1 if any(r.status == "fail" for r in results) else 0)


if __name__ == "__main__":
    main()
