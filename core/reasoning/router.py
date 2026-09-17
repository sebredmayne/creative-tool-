"""Decides whether a query needs semantic retrieval, structured data analysis, or both.

V0 uses simple keyword heuristics rather than an LLM call, so routing works
without any API key configured and is easy to unit test. This can be
upgraded to an LLM-based classifier later without changing how callers use
`decide_route`.
"""
from dataclasses import dataclass

_STRUCTURED_KEYWORDS = (
    "average", "mean", "median", "total", "sum", "count", "how many",
    "percent", "%", "trend", "compare", "correlation", "growth", "rate",
    "highest", "lowest", "top ", "bottom ",
)


@dataclass
class Route:
    use_semantic: bool
    use_structured: bool


def decide_route(query: str, structured_data_available: bool) -> Route:
    if not structured_data_available:
        return Route(use_semantic=True, use_structured=False)

    query_lower = query.lower()
    looks_structured = any(keyword in query_lower for keyword in _STRUCTURED_KEYWORDS)
    return Route(use_semantic=True, use_structured=looks_structured)
