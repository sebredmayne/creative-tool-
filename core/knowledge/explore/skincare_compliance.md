# GlowLabs - Compliance & Claim Rules (Non-Negotiable)

These rules are loaded deterministically into every GlowLabs request's system prompt - they do
not depend on vector search ranking them highly, and they are never optional or "one input
among several." If anything else in the retrieved knowledge or the user's request conflicts
with a rule here, this file wins.

Cosmetics claims in India sit under different rules than drugs: a cosmetic can describe
cleansing, beautifying, or altering appearance, but the moment a claim implies curing,
treating, or preventing a medical condition (acne as a disease, eczema, fungal infections), it
risks being treated as an unapproved drug claim rather than a cosmetic one. That line has to be
enforced by the content system, not caught after the fact.

| Flag | Fails if | Fix |
|---|---|---|
| S1 | "Cures," "treats," "eliminates," or "prevents" applied to a skin condition (acne, eczema, pigmentation as a medical term) | Reframe to appearance-based language: "helps improve the appearance of," "helps reduce the look of" |
| S2 | "Clinically proven," "dermatologically tested," or "100% natural" without an actual test/certification behind it | Remove, or replace with the specific verifiable fact (e.g. name the actual ingredient study if one genuinely exists) |
| S3 | An SPF number, percentage actives (e.g. "5% Niacinamide"), or "broad-spectrum" claim not backed by the product's actual tested formulation | Only state values that match the real formulation and testing |
| S4 | A before/after claim with a specific timeline ("clear skin in 7 days") and no "results may vary" framing | Remove the fixed timeline; if a timeframe is used at all, pair it with a visible disclaimer |
| S5 | Body-shaming or skin-shaming language ("ugly," "flawless skin," implying current skin is a problem to be fixed) | Reframe around care and confidence, never around a flaw to be ashamed of |
| S6 | A comparison naming or visually implying a specific competitor product | Remove - unnamed structural contrast only |

Response protocol: **Proceed**, **Soft warning** (flag for the brand/regulatory team -
especially any ingredient-percentage or SPF claim), or **Hard stop**. Any claim that reads as
diagnosing or curing a skin condition should default to Hard stop, not Soft warning - that's
the line between a cosmetic claim and a drug claim, and it's not a judgment call to make
loosely.
