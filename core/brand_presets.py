"""The three categories this demo is deepest on.

Picking one auto-loads that brand's own sample data + pre-fills its context -
so users always know what's actually well-supported, rather than typing
anything and guessing.

These are BrandProfile instances - the same shape a marketer's own custom brand ends up in
(core/custom_brands.py) once created through the "Add your brand" flow, so the rest of the app
never needs to know whether a given brand is a preset or user-created. voice_and_tone/
positioning/personas are left blank here since their content already lives in
core/knowledge/explore/*.md and is retrieved from there either way; compliance_rules is left
blank too since core.knowledge.loader.load_compliance_rules() reads the static
<slug>_compliance.md file for these instead of this field (custom brands use this field
directly, since they have no such file).
"""
from core.models import BrandProfile

BRAND_PRESETS: dict[str, BrandProfile] = {
    "sproutmix": BrandProfile(
        slug="sproutmix",
        label="Kids Nutrition",
        brand_name="Sunny Sprout / SproutMix",
        description="Milk-mix nutrition brand for kids (SM2 / SM7 / SM13)",
        sample_files=["sproutmix_reviews.csv", "sproutmix_sales.xlsx"],
        example_prompts=[
            {
                "category": "Reel scripts",
                "prompt": "Give me 3 Reel scripts about why kids reject milk mix, and how we solve it",
            },
            {"category": "Ad concepts", "prompt": "Give me ad angles based on our top review complaints"},
            {"category": "Campaign", "prompt": "Give me a back-to-school campaign idea"},
        ],
        context={
            "brand": "Sunny Sprout",
            "product": "SproutMix (SM2/SM7/SM13 milk mix)",
            "customer": "Parents of kids aged 2-13",
            "category": "Kids nutrition",
            "objective": "Acquisition and retention",
        },
    ),
    "skincare": BrandProfile(
        slug="skincare",
        label="Skincare & Personal Care",
        brand_name="GlowLabs",
        description="Vitamin C Face Wash, Niacinamide Serum, Sunscreen, plus a wider hair/body/gummies range",
        sample_files=[
            "skincare_reviews.csv",
            "skincare_sales.xlsx",
            "personalcare_creative_tests.csv",
            "personalcare_narrative_angles.csv",
        ],
        example_prompts=[
            {"category": "Reel scripts", "prompt": "Give me 3 Instagram Reel scripts for a summer skincare launch"},
            {"category": "Ad concepts", "prompt": "Give me ad angles based on our top review complaints"},
            {"category": "Campaign", "prompt": "Give me a routine-building campaign idea for first-time buyers"},
        ],
        context={
            "brand": "GlowLabs",
            "product": "Vitamin C Face Wash / Niacinamide Serum / Sunscreen SPF50",
            "customer": "Women 20-35 building a skincare routine",
            "category": "Skincare",
            "objective": "Acquisition",
        },
    ),
    "flexwear": BrandProfile(
        slug="flexwear",
        label="Fitness Apparel",
        brand_name="FlexWear",
        description="Activewear - Core/Power/Recovery leggings and joggers",
        sample_files=["flexwear_reviews.csv", "flexwear_sales.xlsx"],
        example_prompts=[
            {"category": "Reel scripts", "prompt": "Give me 3 Reel scripts about why our leggings actually fit"},
            {"category": "Ad concepts", "prompt": "Give me ad angles based on our top review complaints"},
            {"category": "Campaign", "prompt": "Give me a size-inclusivity campaign idea"},
        ],
        context={
            "brand": "FlexWear",
            "product": "Core/Power/Recovery activewear line",
            "customer": "Women 24-40, mixed fitness levels",
            "category": "Fitness apparel",
            "objective": "Acquisition and repeat purchase",
        },
    ),
}
