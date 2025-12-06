"""
Comprehensive Kalshi API Integration Service.
Maps Kalshi SDK categories to direct HTTP requests using core.security.kalshi_request.

Reference: https://docs.kalshi.com/trade-api/v2
"""

import logging
from typing import Any, Dict, Optional

from core.security import kalshi_request
from services.cache import get_events_cache, get_market_cache

logger = logging.getLogger(__name__)


# --- Portfolio API ---
# Endpoints for managing user funds, orders, and positions.

def get_portfolio_balance() -> Dict[str, Any]:
    """Retrieve the user's current balance."""
    return kalshi_request("GET", "/portfolio/balance").json()


def get_positions(
    limit: int = 100,
    cursor: Optional[str] = None,
    settlement_status: Optional[str] = None,
    ticker: Optional[str] = None
) -> Dict[str, Any]:
    """Get current positions."""
    params = {
        "limit": limit,
        "cursor": cursor,
        "settlement_status": settlement_status,
        "ticker": ticker
    }
    return kalshi_request("GET", "/portfolio/positions", params=params).json()


def get_orders(
    status: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 100,
    cursor: Optional[str] = None
) -> Dict[str, Any]:
    """List orders."""
    params = {
        "status": status,
        "ticker": ticker,
        "limit": limit,
        "cursor": cursor
    }
    return kalshi_request("GET", "/portfolio/orders", params=params).json()


def create_order(
    ticker: str,
    action: str,
    type: str,
    side: str,
    count: int,
    yes_price: Optional[int] = None,
    no_price: Optional[int] = None,
    expiration_ts: Optional[int] = None,
    client_order_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place a new order.
    action: 'buy' or 'sell'
    type: 'limit' or 'market'
    side: 'yes' or 'no'
    """
    payload = {
        "ticker": ticker,
        "action": action,
        "type": type,
        "side": side,
        "count": count,
        "yes_price": yes_price,
        "no_price": no_price,
        "expiration_ts": expiration_ts,
        "client_order_id": client_order_id
    }
    return kalshi_request("POST", "/portfolio/orders", json=payload).json()


def cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel a specific order."""
    return kalshi_request("DELETE", f"/portfolio/orders/{order_id}").json()


def get_fills(
    ticker: Optional[str] = None,
    order_id: Optional[str] = None,
    limit: int = 100,
    cursor: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve fill history."""
    params = {
        "ticker": ticker,
        "order_id": order_id,
        "limit": limit,
        "cursor": cursor
    }
    return kalshi_request("GET", "/portfolio/fills", params=params).json()


# --- Markets API ---

def get_markets(
    limit: int = 100,
    cursor: Optional[str] = None,
    status: Optional[str] = "open",
    series_ticker: Optional[str] = None,
    event_ticker: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """List markets with optional caching."""
    cache_key = f"markets:{limit}:{cursor}:{status}:{series_ticker}:{event_ticker}"

    if use_cache:
        cache = get_market_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for markets list")
            return cached

    params = {
        "limit": limit,
        "cursor": cursor,
        "status": status,
        "series_ticker": series_ticker,
        "event_ticker": event_ticker
    }
    result = kalshi_request("GET", "/markets", params=params).json()

    if use_cache:
        cache = get_market_cache()
        cache.set(cache_key, result, ttl=60.0)

    return result





def get_market(ticker: str, use_cache: bool = True) -> Dict[str, Any]:
    """Get details for a single market with optional caching."""
    cache_key = f"market:{ticker}"

    if use_cache:
        cache = get_market_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for market %s", ticker)
            return cached

    result = kalshi_request("GET", f"/markets/{ticker}").json()

    if use_cache:
        cache = get_market_cache()
        cache.set(cache_key, result, ttl=30.0)

    return result






def get_market_candlesticks(
    ticker: str,
    series_ticker: str,
    period_interval: int,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get OHLCV candlestick data.
    period_interval: 1, 60, 3600, 86400 (minutes/hours in seconds)
    """
    params = {
        "series_ticker": series_ticker,
        "period_interval": period_interval,
        "start_ts": start_ts,
        "end_ts": end_ts
    }
    return kalshi_request("GET", f"/markets/{ticker}/candlesticks", params=params).json()





def get_market_trades(
    ticker: str,
    limit: int = 100,
    cursor: Optional[str] = None
) -> Dict[str, Any]:
    """Get recent public trades for a market."""
    params = {"limit": limit, "cursor": cursor}
    return kalshi_request("GET", f"/markets/{ticker}/trades", params=params).json()





# --- Events API ---

# Groups of markets (e.g., "Will result X happen?").



def get_events(
    limit: int = 100,
    cursor: Optional[str] = None,
    status: Optional[str] = None,
    series_ticker: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """List events with optional caching."""
    cache_key = f"events:{limit}:{cursor}:{status}:{series_ticker}"

    if use_cache:
        cache = get_events_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for events list")
            return cached

    params = {
        "limit": limit,
        "cursor": cursor,
        "status": status,
        "series_ticker": series_ticker
    }
    result = kalshi_request("GET", "/events", params=params).json()

    if use_cache:
        cache = get_events_cache()
        cache.set(cache_key, result, ttl=300.0)  # 5 minutes

    return result


def get_event(event_ticker: str, use_cache: bool = True) -> Dict[str, Any]:
    """Get a specific event with optional caching."""
    cache_key = f"event:{event_ticker}"

    if use_cache:
        cache = get_events_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for event %s", event_ticker)
            return cached

    result = kalshi_request("GET", f"/events/{event_ticker}").json()

    if use_cache:
        cache = get_events_cache()
        cache.set(cache_key, result, ttl=60.0)  # 1 minute

    return result


# --- Series API ---
# High-level groupings (e.g., "Weekly Jobless Claims").

def get_series(series_ticker: str) -> Dict[str, Any]:
    """Get details of a series."""
    return kalshi_request("GET", f"/series/{series_ticker}").json()


# --- Multivariate Collections API ---
# Collections of related markets.

def get_multivariate_collection(collection_ticker: str) -> Dict[str, Any]:
    """Get a multivariate collection by ticker."""
    return kalshi_request("GET", f"/multivariate_collections/{collection_ticker}").json()


# --- Structured Targets API ---
# Specific target-based market structures.

def get_structured_targets(
    limit: int = 100,
    cursor: Optional[str] = None
) -> Dict[str, Any]:
    """List structured targets."""
    # Assuming standard pattern; specific endpoint depends on actual API spec
    return kalshi_request("GET", "/structured_targets", params={"limit": limit, "cursor": cursor}).json()


# --- Api Keys API ---
# Programmatic access key management.

def list_api_keys() -> Dict[str, Any]:
    """List API keys associated with the account."""
    return kalshi_request("GET", "/portfolio/keys").json()


def create_api_key(name: str, pem_public_key: str) -> Dict[str, Any]:
    """Create a new API key."""
    payload = {"name": name, "public_key": pem_public_key}
    return kalshi_request("POST", "/portfolio/keys", json=payload).json()


# --- Communications API ---
# User notifications and announcements.

def get_notifications(
    limit: int = 50,
    cursor: Optional[str] = None
) -> Dict[str, Any]:
    """Get user notifications."""
    # Common convention for user-facing notifications in Kalshi V2
    return kalshi_request("GET", "/portfolio/notifications", params={"limit": limit, "cursor": cursor}).json()


# --- Milestones API ---
# User achievements or volume tiers.

def get_milestones() -> Dict[str, Any]:
    """Get user milestones."""
    # Path inferred from SDK structure; typically under portfolio or users
    return kalshi_request("GET", "/portfolio/milestones").json()
