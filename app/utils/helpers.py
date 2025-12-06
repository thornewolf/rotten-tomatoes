"""
General helper functions for the application.
"""

from typing import List


def parse_query_terms(raw_queries: List[str]) -> List[str]:
    """
    Split and trim query terms from repeated params, commas, or newlines.

    Handles multiple input formats:
    - Multiple query parameters: ?q=foo&q=bar
    - Comma-separated: ?q=foo,bar
    - Newline-separated: ?q=foo%0Abar

    Args:
        raw_queries: List of raw query strings from URL parameters

    Returns:
        List of unique, cleaned query terms
    """
    terms: List[str] = []
    for raw in raw_queries:
        if not raw:
            continue
        # Split on commas or newlines
        for candidate in raw.replace(",", "\n").split("\n"):
            cleaned = candidate.strip()
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
    return terms
