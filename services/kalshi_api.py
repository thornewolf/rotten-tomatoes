import logging
from typing import Any, Dict, List, Optional

from app.schemas import MarketSummary
from services import kalshi_service

logger = logging.getLogger(__name__)


def _normalize_price(value: Any) -> Optional[float]:
    """Convert Kalshi price (cents) to dollars, handling already-normalized floats."""
    if value is None:
        return None
    price = float(value)
    return round(price / 100, 4) if price > 1 else round(price, 4)


def _build_market_summary(market: Dict[str, Any]) -> MarketSummary:
    """Build a MarketSummary from a single market dict."""
    if not isinstance(market, dict):
        raise ValueError(f"Market payload must be a dict, got {type(market)}")
    ticker = market.get("ticker") or market.get("event_ticker") or market.get("id")
    if not ticker:
        raise ValueError("Market summary is missing a ticker/id")
    link_slug_source = market.get("event_ticker") or ticker
    link_slug = str(link_slug_source).split("-")[0]
    return MarketSummary(
        ticker=ticker,
        title=market.get("title"),
        category=market.get("category"),
        status=market.get("status"),
        series_ticker=market.get("series_ticker") or market.get("series"),
        event_ticker=market.get("event_ticker") or market.get("event"),
        volume=market.get("volume_24h") or market.get("volume"),
        open_interest=market.get("open_interest") or market.get("oi"),
        close_time=market.get("close_time") or market.get("close_ts"),
        yes_bid=_normalize_price(market.get("yes_bid")),
        yes_ask=_normalize_price(market.get("yes_ask")),
        no_bid=_normalize_price(market.get("no_bid")),
        no_ask=_normalize_price(market.get("no_ask")),
        link=f"https://kalshi.com/markets/{link_slug}",
    )


def _estimate_score_from_markets(markets: List[Dict[str, Any]]) -> Optional[float]:
    """
    Estimate the expected RT score from market prices.

    Each market is "Above X" with a yes price = P(score > X).
    We extract (strike, probability) pairs and compute the expected value.

    The probability that score is in range (X_i, X_{i+1}] is P(>X_i) - P(>X_{i+1}).
    Expected value = sum of midpoint * probability for each bucket.
    """
    if not markets:
        return None

    # Extract (strike, mid_price) pairs, sorted by strike
    strike_probs = []
    for m in markets:
        strike = m.get("floor_strike")
        if strike is None:
            continue
        # Use mid price between yes_bid and yes_ask, or last_price as fallback
        yes_bid = m.get("yes_bid")
        yes_ask = m.get("yes_ask")
        if yes_bid is not None and yes_ask is not None:
            mid_price = (yes_bid + yes_ask) / 2
        else:
            mid_price = m.get("last_price", 0) or 0
        # Convert from cents (0-100) to probability (0-1)
        prob = mid_price / 100
        strike_probs.append((strike, prob))

    if not strike_probs:
        return None

    # Sort by strike ascending
    strike_probs.sort(key=lambda x: x[0])

    # Calculate expected value using the probability distribution
    # P(score > X) decreases as X increases
    # P(score in (X_i, X_{i+1}]) = P(>X_i) - P(>X_{i+1})
    expected_value = 0.0

    # Handle the range below the lowest strike: [0, first_strike]
    first_strike, first_prob = strike_probs[0]
    prob_below_first = 1.0 - first_prob
    if prob_below_first > 0:
        midpoint = first_strike / 2  # midpoint of [0, first_strike]
        expected_value += midpoint * prob_below_first

    # Handle ranges between consecutive strikes
    for i in range(len(strike_probs) - 1):
        strike_low, prob_low = strike_probs[i]
        strike_high, prob_high = strike_probs[i + 1]
        prob_in_range = prob_low - prob_high
        if prob_in_range > 0:
            midpoint = (strike_low + strike_high) / 2
            expected_value += midpoint * prob_in_range

    # Handle the range above the highest strike: [last_strike, 100]
    last_strike, last_prob = strike_probs[-1]
    if last_prob > 0:
        midpoint = (last_strike + 100) / 2
        expected_value += midpoint * last_prob

    return round(expected_value, 1)


def _build_event_summary(event: Dict[str, Any], markets: List[Dict[str, Any]]) -> MarketSummary:
    """Build a MarketSummary from an event and its child markets, aggregating data."""
    if not isinstance(event, dict):
        raise ValueError(f"Event payload must be a dict, got {type(event)}")
    ticker = event.get("event_ticker") or event.get("ticker")
    if not ticker:
        raise ValueError("Event is missing event_ticker")

    # Aggregate volume and open_interest from all child markets
    total_volume = sum(m.get("volume_24h", 0) or m.get("volume", 0) or 0 for m in markets)
    total_open_interest = sum(m.get("open_interest", 0) or 0 for m in markets)

    # Get close_time from the first market (they should all be the same)
    close_time = markets[0].get("close_time") if markets else None

    # Get status from markets (use first active, or first overall)
    status = None
    for m in markets:
        if m.get("status") == "active":
            status = "active"
            break
        if not status:
            status = m.get("status")

    # Calculate estimated score from market prices
    estimated_score = _estimate_score_from_markets(markets)

    return MarketSummary(
        ticker=ticker,
        title=event.get("title"),
        category=event.get("category"),
        status=status,
        series_ticker=event.get("series_ticker"),
        event_ticker=ticker,
        volume=total_volume,
        open_interest=total_open_interest,
        close_time=close_time,
        yes_bid=None,  # Not meaningful at event level
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        link=f"https://kalshi.com/markets/{ticker}",
        estimated_score=estimated_score,
    )


def fetch_single_market_summary(ticker: str) -> MarketSummary:
    """Get a single market/event summary using the events API."""
    if not ticker:
        raise ValueError("Ticker is required for single market summary")

    event_response = kalshi_service.get_event(ticker)
    if not event_response:
        raise ValueError(f"Event not found for {ticker}")
    if not isinstance(event_response, dict):
        raise ValueError(f"Unexpected event response type {type(event_response)} for {ticker}")

    event = event_response.get("event", {})
    markets = event_response.get("markets", [])

    return _build_event_summary(event, markets)


def fetch_market_summaries(limit: int = 20) -> List[MarketSummary]:
    """Fetch events for summary display."""
    data = kalshi_service.get_events(limit=limit)
    if not isinstance(data, dict):
        raise ValueError(f"Events response must be a dict, got {type(data)}")
    events = data.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"'events' field must be a list, got {type(events)}")

    summaries: List[MarketSummary] = []
    for event in events[:limit]:
        summaries.append(_build_market_summary(event))
    summaries.sort(key=lambda m: m.volume or 0, reverse=True)
    logger.info("Successfully fetched %s event summaries from Kalshi API", len(summaries))
    return summaries


def search_market(query: str) -> MarketSummary | None:
    """
    Search for a market by ticker or title and return a summary.
    """
    results = search_markets([query])
    return results.get(query)


def search_markets(
    queries: List[str],
    depth: int = 3,
    search_limit: int = 1000,
) -> Dict[str, MarketSummary | None]:
    """
    Batch search for events by ticker or title; returns map of query->snapshot or None.
    """
    if not queries:
        return {}

    data = kalshi_service.get_events(limit=search_limit)
    if not isinstance(data, dict):
        raise ValueError(f"Events response must be a dict, got {type(data)}")
    events = data.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"'events' field must be a list, got {type(events)}")
    results: Dict[str, MarketSummary | None] = {}

    for query in queries:
        if not query:
            continue
        query_lower = query.lower()
        match = next(
            (
                event
                for event in events
                if query_lower in str(event.get("event_ticker", "")).lower()
                or query_lower in str(event.get("title", "")).lower()
            ),
            None,
        )
        if not match:
            results[query] = None
            continue

        results[query] = _build_market_summary(match)

    return results
