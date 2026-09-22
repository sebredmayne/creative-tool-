# Sample Brand Deep-Dive: GlowLabs - Brand Voice & Compliance Gate

A worked example of a skincare D2C brand, at the same depth as the SproutMix and FlexWear examples - illustrative/fictional, representative of real Indian skincare D2C brands. These products match the ones in `sample_data/reviews.csv` and `sample_data/sales.xlsx` (Vitamin C Face Wash, Niacinamide Serum, Sunscreen SPF50), so this file gives that sample data an actual brand identity to be grounded against.

## Brand foundation & voice

GlowLabs is a clean-label, ingredient-transparent skincare brand. Brand purpose: make effective skincare legible - the promise is confidence in what's in the bottle, not a guaranteed transformation. Tagline: "Know what you put on your skin."

Archetype: Sage ("the truth will help you") layered with Caregiver (gentle, non-judgmental about skin concerns). Tone - do: clear ingredient explanations, calm and factual, real skin textures shown (not over-retouched). Tone - avoid: shame-based hooks ("get rid of ugly acne"), miracle-cure language, before/after framing without a "results may vary" reality check.

Pricing & positioning: mid-premium, framed around ingredient concentration and formulation quality rather than discount. Avoid manufactured urgency on evergreen SKUs.

## Product line at a glance

Three core SKUs (matching the sample data):

| | Vitamin C Face Wash | Niacinamide Serum | Sunscreen SPF50 |
|---|---|---|---|
| Core concern | Dullness, uneven tone | Pores, oil control | Sun protection, daily wear |
| Key ingredient | Vitamin C (stabilized) | 5% Niacinamide | Broad-spectrum SPF50 filters |
| Approved claim shape | "Helps improve the appearance of dullness with regular use" | "Helps minimize the appearance of enlarged pores" | "Broad-spectrum SPF50, tested for the labeled protection level" |
| Never claim | "Cures" or "treats" dullness/acne (medical/drug-territory language) | "Eliminates" pores (physically impossible; softens to "appearance of") | An SPF number not backed by actual lab testing |

## The compliance gate - non-negotiable

Cosmetics claims in India sit under different rules than drugs: a cosmetic can describe cleansing, beautifying, or altering appearance, but the moment a claim implies curing, treating, or preventing a medical condition (acne as a disease, eczema, fungal infections), it risks being treated as an unapproved drug claim rather than a cosmetic one. That line has to be enforced by the content system, not caught after the fact.

| Flag | Fails if | Fix |
|---|---|---|
| S1 | "Cures," "treats," "eliminates," or "prevents" applied to a skin condition (acne, eczema, pigmentation as a medical term) | Reframe to appearance-based language: "helps improve the appearance of," "helps reduce the look of" |
| S2 | "Clinically proven," "dermatologically tested," or "100% natural" without an actual test/certification behind it | Remove, or replace with the specific verifiable fact (e.g. name the actual ingredient study if one genuinely exists) |
| S3 | An SPF number, percentage actives (e.g. "5% Niacinamide"), or "broad-spectrum" claim not backed by the product's actual tested formulation | Only state values that match the real formulation and testing |
| S4 | A before/after claim with a specific timeline ("clear skin in 7 days") and no "results may vary" framing | Remove the fixed timeline; if a timeframe is used at all, pair it with a visible disclaimer |
| S5 | Body-shaming or skin-shaming language ("ugly," "flawless skin," implying current skin is a problem to be fixed) | Reframe around care and confidence, never around a flaw to be ashamed of |
| S6 | A comparison naming or visually implying a specific competitor product | Remove - unnamed structural contrast only |

Response protocol: **Proceed**, **Soft warning** (flag for the brand/regulatory team - especially any ingredient-percentage or SPF claim), or **Hard stop**. Any claim that reads as diagnosing or curing a skin condition should default to Hard stop, not Soft warning - that's the line between a cosmetic claim and a drug claim, and it's not a judgment call to make loosely.
