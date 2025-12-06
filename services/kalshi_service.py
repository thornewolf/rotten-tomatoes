"""
Comprehensive Kalshi API Integration Service.
Maps Kalshi SDK categories to direct HTTP requests using core.security.kalshi_request.

Reference: https://docs.kalshi.com/trade-api/v2
"""

from typing import Any, Dict, Optional

from core.security import kalshi_request


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
    # Filter None values
    payload = {k: v for k, v in payload.items() if v is not None}
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
# Public data about markets, orderbooks, and trades.

def get_markets(
    limit: int = 100,
    cursor: Optional[str] = None,
    status: Optional[str] = "open",
    series_ticker: Optional[str] = None,
    event_ticker: Optional[str] = None
) -> Dict[str, Any]:
    """List markets."""
    params = {
        "limit": limit,
        "cursor": cursor,
        "status": status,
        "series_ticker": series_ticker,
        "event_ticker": event_ticker
    }
    return kalshi_request("GET", "/markets", params=params).json()


def get_market(ticker: str) -> Dict[str, Any]:
    """Get details for a single market."""
    return kalshi_request("GET", f"/markets/{ticker}").json()


def get_market_orderbook(ticker: str, depth: int = 25) -> Dict[str, Any]:
    """Get the orderbook for a market."""
    return kalshi_request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth}).json()


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
    series_ticker: Optional[str] = None
) -> Dict[str, Any]:
    """List events."""
    params = {
        "limit": limit,
        "cursor": cursor,
        "status": status,
        "series_ticker": series_ticker
    }
    return kalshi_request("GET", "/events", params=params).json()


def get_event(event_ticker: str) -> Dict[str, Any]:
    """Get a specific event."""
    return kalshi_request("GET", f"/events/{event_ticker}").json()


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
