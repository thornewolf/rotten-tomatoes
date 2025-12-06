import logging
from typing import Any, Dict, List, Optional

from services import kalshi_service

logger = logging.getLogger(__name__)


def _normalize_price(value: Any) -> Optional[float]:
    """Convert Kalshi price (cents) to dollars, handling already-normalized floats."""
    if value is None:
        return None
    price = float(value)
    return round(price / 100, 4) if price > 1 else round(price, 4)


def _parse_levels(levels: Any, depth: int, *, side: str, book_side: str) -> List[Dict[str, Any]]:
    """Normalize orderbook levels into dictionaries with price/size keys. Raise on unexpected shapes."""
    if levels is None:
        return []
    if not isinstance(levels, (list, tuple)):
        raise ValueError(f"Orderbook levels for {side} {book_side} must be list/tuple, got {type(levels)}")

    parsed: List[Dict[str, Any]] = []
    for idx, level in enumerate(levels):
        price = None
        size = None
        if isinstance(level, (list, tuple)):
            price = level[0] if len(level) > 0 else None
            size = level[1] if len(level) > 1 else None
        elif isinstance(level, dict):
            price = level.get("price")
            size = level.get("size") or level.get("count") or level.get("qty")
        else:
            raise ValueError(f"Level {idx} for {side} {book_side} must be list/tuple/dict, got {type(level)}")

        normalized_price = _normalize_price(price)
        if normalized_price is None:
            raise ValueError(f"Missing price in level {idx} for {side} {book_side}")
        normalized_size = int(size) if size is not None else None
        parsed.append({"price": normalized_price, "size": normalized_size})

        if len(parsed) >= depth:
            break
    return parsed


def _top_bid(levels: List[Dict[str, Any]]) -> Optional[float]:
    bids = [level["price"] for level in levels if level.get("price") is not None]
    return max(bids) if bids else None


def _top_ask(levels: List[Dict[str, Any]]) -> Optional[float]:
    asks = [level["price"] for level in levels if level.get("price") is not None]
    return min(asks) if asks else None


def _build_orderbook_snapshot(orderbook_raw: Dict[str, Any], depth: int) -> Dict[str, Any]:
    if not isinstance(orderbook_raw, dict):
        raise ValueError(f"Orderbook payload must be a dict, got {type(orderbook_raw)}")

    yes_side = orderbook_raw.get("yes")
    no_side = orderbook_raw.get("no")
    if not isinstance(yes_side, dict) or not isinstance(no_side, dict):
        raise ValueError("Orderbook must contain dict sides for 'yes' and 'no'")

    yes_bids = _parse_levels(yes_side.get("bids") or yes_side.get("bid"), depth, side="yes", book_side="bids")
    yes_asks = _parse_levels(yes_side.get("asks") or yes_side.get("ask"), depth, side="yes", book_side="asks")
    no_bids = _parse_levels(no_side.get("bids") or no_side.get("bid"), depth, side="no", book_side="bids")
    no_asks = _parse_levels(no_side.get("asks") or no_side.get("ask"), depth, side="no", book_side="asks")

    orderbook = {
        "yes": {"bids": yes_bids, "asks": yes_asks},
        "no": {"bids": no_bids, "asks": no_asks},
    }

    top_of_book = {
        "yes": {"bid": _top_bid(yes_bids), "ask": _top_ask(yes_asks)},
        "no": {"bid": _top_bid(no_bids), "ask": _top_ask(no_asks)},
    }
    return {"orderbook": orderbook, "top_of_book": top_of_book}


def _build_market_snapshot(market: Dict[str, Any], orderbook_snapshot: Dict[str, Any], depth: int) -> Dict[str, Any]:
    if not isinstance(market, dict):
        raise ValueError(f"Market payload must be a dict, got {type(market)}")

    ticker = market.get("ticker") or market.get("id")
    if not ticker:
        raise ValueError("Market is missing a ticker/id")

    if not isinstance(orderbook_snapshot, dict):
        raise ValueError(f"Orderbook snapshot must be a dict, got {type(orderbook_snapshot)}")
    if "top_of_book" not in orderbook_snapshot or "orderbook" not in orderbook_snapshot:
        raise ValueError("Orderbook snapshot missing required keys")

    top = orderbook_snapshot["top_of_book"]
    yes_bid = top["yes"]["bid"]
    yes_ask = top["yes"]["ask"]
    no_bid = top["no"]["bid"]
    no_ask = top["no"]["ask"]

    snapshot = {
        "ticker": ticker,
        "title": market.get("title"),
        "category": market.get("category"),
        "status": market.get("status"),
        "series_ticker": market.get("series_ticker") or market.get("series"),
        "event_ticker": market.get("event_ticker") or market.get("event"),
        "volume": market.get("volume_24h") or market.get("volume"),
        "open_interest": market.get("open_interest") or market.get("oi"),
        "close_time": market.get("close_time") or market.get("close_ts"),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "orderbook": orderbook_snapshot["orderbook"],
        "top_of_book": top,
        "orderbook_depth": depth,
        "link": f"https://kalshi.com/markets/{ticker}" if ticker else None,
    }
    return snapshot


def get_orderbook_snapshot(ticker: str, depth: int = 5) -> Dict[str, Any]:
    """Fetch and normalize an orderbook for a ticker."""
    raw_response = kalshi_service.get_market_orderbook(ticker, depth=depth)
    if not isinstance(raw_response, dict):
        raise ValueError(f"Orderbook response must be a dict, got {type(raw_response)}")

    orderbook_payload = raw_response.get("orderbook") or raw_response
    if not isinstance(orderbook_payload, dict):
        raise ValueError(f"Orderbook payload must be a dict, got {type(orderbook_payload)}")

    orderbook_snapshot = _build_orderbook_snapshot(orderbook_payload, depth)
    logger.info("Fetched orderbook for %s (depth=%s)", ticker, depth)
    return orderbook_snapshot


def fetch_single_market_snapshot(ticker: str, depth: int = 5) -> Optional[Dict[str, Any]]:
    """Get a full snapshot (market summary + orderbook) for one ticker."""
    if not ticker:
        raise ValueError("Ticker is required for single market snapshot")
    market_response = kalshi_service.get_market(ticker)
    if not market_response:
        raise ValueError(f"Market response missing for {ticker}")
    if isinstance(market_response, dict):
        market = market_response.get("market") or market_response
    elif isinstance(market_response, list) and market_response:
        market = market_response[0]
    else:
        raise ValueError(f"Unexpected market response type {type(market_response)} for {ticker}")
    orderbook_snapshot = get_orderbook_snapshot(ticker, depth=depth)
    return _build_market_snapshot(market, orderbook_snapshot, depth)


def fetch_markets_from_kalshi(limit: int = 20, depth: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch markets from Kalshi API using proper request signing and enrich with orderbooks.
    """
    data = kalshi_service.get_markets(limit=limit)
    if not isinstance(data, dict):
        raise ValueError(f"Markets response must be a dict, got {type(data)}")
    markets = data.get("markets", [])
    if not isinstance(markets, list):
        raise ValueError(f"'markets' field must be a list, got {type(markets)}")
    snapshots: List[Dict[str, Any]] = []
    for i, market in enumerate(markets[:limit]):
        ticker = market.get("ticker") or market.get("id")
        if not ticker:
            raise ValueError(f"Market entry missing ticker/id at index {i}")
        orderbook_snapshot = get_orderbook_snapshot(ticker, depth=depth)
        snapshots.append(_build_market_snapshot(market, orderbook_snapshot, depth))
    logger.info("Successfully fetched %s markets with orderbooks from Kalshi API", len(snapshots))
    return snapshots


def search_market(query: str) -> Dict[str, Any] | None:
    """
    Search for a market by ticker or title and return a snapshot with orderbook.
    """
    results = search_markets([query], depth=3)
    return results.get(query)


def search_markets(
    queries: List[str],
    depth: int = 3,
    search_limit: int = 150,
) -> Dict[str, Dict[str, Any] | None]:
    """
    Batch search for markets by ticker or title; returns map of query->snapshot or None.
    """
    if not queries:
        return {}

    data = kalshi_service.get_markets(limit=search_limit)
    if not isinstance(data, dict):
        raise ValueError(f"Markets response must be a dict, got {type(data)}")
    markets = data.get("markets", [])
    if not isinstance(markets, list):
        raise ValueError(f"'markets' field must be a list, got {type(markets)}")
    results: Dict[str, Dict[str, Any] | None] = {}

    for query in queries:
        if not query:
            continue
        query_lower = query.lower()
        match = next(
            (
                market
                for market in markets
                if query_lower in str(market.get("ticker", "")).lower()
                or query_lower in str(market.get("title", "")).lower()
            ),
            None,
        )
        if not match:
            results[query] = None
            continue

        ticker = match.get("ticker") or match.get("id")
        orderbook_snapshot = get_orderbook_snapshot(ticker, depth=depth)
        results[query] = _build_market_snapshot(match, orderbook_snapshot, depth)

    return results
