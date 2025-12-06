import logging
from typing import Any, Dict, List

from core.security import kalshi_request

logger = logging.getLogger(__name__)


def fetch_markets_from_kalshi(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch markets from Kalshi API using proper request signing.
    """
    response = kalshi_request("GET", f"/markets?status=open&limit={limit}", timeout=10)
    response.raise_for_status()  # Raise an exception for bad status codes

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


def get_market_orderbook(ticker: str, depth: int = 0) -> Dict[str, Any]:
    """
    Fetch orderbook for a specific market from Kalshi API.
    """
    response = kalshi_request("GET", f"/markets/{ticker}/orderbook?depth={depth}", timeout=10)
    response.raise_for_status()  # Raise an exception for bad status codes

    data = response.json()
    orderbook = data.get("orderbook", {})
    logger.info("Successfully fetched orderbook for %s", ticker)
    return orderbook
