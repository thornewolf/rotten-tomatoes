import logging
from typing import Any, Dict, List

from core.security import kalshi_request

logger = logging.getLogger(__name__)


def fetch_markets_from_kalshi(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch markets from Kalshi API using proper request signing.
    Falls back to mock data if API is not available or not configured.
    """
    try:
        response = kalshi_request("GET", f"/markets?status=open&limit={limit}", timeout=10)

        if response.status_code == 200:
            data = response.json()
            markets = data.get("markets", [])
            logger.info("Successfully fetched %s markets from Kalshi API", len(markets))

            return [
                {
                    "id": market.get("ticker", f"market_{i}"),
                    "title": market.get("title", f"Market {i}"),
                    "category": market.get("category", "General"),
                    "status": market.get("status", "open"),
                    "yes_price": market.get("yes_bid", 0.5),
                }
                for i, market in enumerate(markets[:limit])
            ]

        logger.warning("Kalshi API error: %s, using mock data", response.status_code)
        return get_mock_markets()

    except Exception as exc:  # noqa: BLE001
        logger.error("Error fetching from Kalshi API: %s, using mock data", exc)
        return get_mock_markets()


def search_market(query: str) -> Dict[str, Any] | None:
    """
    Search for a market by ticker or title.
    Returns the first match or None.
    """
    # Fetch more markets to increase chance of finding it
    markets = fetch_markets_from_kalshi(limit=100)
    query_lower = query.lower()
    for market in markets:
        if query_lower in market.get("id", "").lower() or query_lower in market.get("title", "").lower():
            return market
    return None


def get_mock_markets() -> List[Dict[str, Any]]:
    """Mock market data for development and testing."""
    return [
        {
            "id": "RT-GLADIATOR2",
            "title": "Rotten Tomatoes: Gladiator 2 > 85%?",
            "category": "Entertainment",
            "status": "open",
            "yes_price": 0.65,
        },
        {
            "id": "RT-WICKED",
            "title": "Rotten Tomatoes: Wicked > 90%?",
            "category": "Entertainment",
            "status": "open",
            "yes_price": 0.25,
        },
    ]
