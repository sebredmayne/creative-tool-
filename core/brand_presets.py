"""The three categories this demo is deepest on.

Picking one auto-loads that brand's own sample data + pre-fills its context -
so users always know what's actually well-supported, rather than typing
anything and guessing. Shared by the Streamlit app and the API backend so
there's one source of truth instead of duplicated brand definitions.
"""

BRAND_PRESETS = {
    "sproutmix": {
        "label": "Kids Nutrition",
        "brand_name": "Sunny Sprout / SproutMix",
        "description": "Milk-mix nutrition brand for kids (SM2 / SM7 / SM13)",
        "sample_files": ["sproutmix_reviews.csv", "sproutmix_sales.xlsx"],
        "example_prompt": "Give me 3 Reel scripts about why kids reject milk mix, and how we solve it",
        "example_prompts": [
            {
                "category": "Reel scripts",
                "prompt": "Give me 3 Reel scripts about why kids reject milk mix, and how we solve it",
            },
            {"category": "Ad concepts", "prompt": "Give me ad angles based on our top review complaints"},
            {"category": "Campaign", "prompt": "Give me a back-to-school campaign idea"},
        ],
        "context": {
            "brand": "Sunny Sprout",
            "product": "SproutMix (SM2/SM7/SM13 milk mix)",
            "customer": "Parents of kids aged 2-13",
            "category": "Kids nutrition",
            "objective": "Acquisition and retention",
        },
    },
    "skincare": {
        "label": "Skincare & Personal Care",
        "brand_name": "GlowLabs",
        "description": "Vitamin C Face Wash, Niacinamide Serum, Sunscreen, plus a wider hair/body/gummies range",
        "sample_files": [
            "reviews.csv",
            "sales.xlsx",
            "personalcare_creative_tests.csv",
            "personalcare_narrative_angles.csv",
        ],
        "example_prompt": "Give me 3 Instagram Reel scripts for a summer skincare launch",
        "example_prompts": [
            {"category": "Reel scripts", "prompt": "Give me 3 Instagram Reel scripts for a summer skincare launch"},
            {"category": "Ad concepts", "prompt": "Give me ad angles based on our top review complaints"},
            {
                "category": "Campaign",
                "prompt": "Give me a routine-building campaign idea for first-time buyers",
            },
        ],
        "context": {
            "brand": "GlowLabs",
            "product": "Vitamin C Face Wash / Niacinamide Serum / Sunscreen SPF50",
            "customer": "Women 20-35 building a skincare routine",
            "category": "Skincare",
            "objective": "Acquisition",
        },
    },
    "flexwear": {
        "label": "Fitness Apparel",
        "brand_name": "FlexWear",
        "description": "Activewear - Core/Power/Recovery leggings and joggers",
        "sample_files": ["flexwear_reviews.csv", "flexwear_sales.xlsx"],
        "example_prompt": "Give me 3 Reel scripts about why our leggings actually fit",
        "example_prompts": [
            {"category": "Reel scripts", "prompt": "Give me 3 Reel scripts about why our leggings actually fit"},
            {"category": "Ad concepts", "prompt": "Give me ad angles based on our top review complaints"},
            {"category": "Campaign", "prompt": "Give me a size-inclusivity campaign idea"},
        ],
        "context": {
            "brand": "FlexWear",
            "product": "Core/Power/Recovery activewear line",
            "customer": "Women 24-40, mixed fitness levels",
            "category": "Fitness apparel",
            "objective": "Acquisition and repeat purchase",
        },
    },
}
