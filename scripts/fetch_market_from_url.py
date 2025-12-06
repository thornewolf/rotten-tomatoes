import argparse
import json
import logging
import sys
from urllib.parse import urlparse

from services.kalshi_api import fetch_single_market_summary
from services import kalshi_service

def extract_ticker_from_url(url: str) -> str:
    """
    Extracts the market ticker from a Kalshi URL.
    Assumption: URL format is like https://kalshi.com/markets/{series}/{slug}/{ticker}
    or https://kalshi.com/markets/{ticker}
    We will take the last path component.
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    parts = path.split("/")
    
    # If the URL ends with a trailing slash, parts[-1] might be empty if we didn't strip.
    # We stripped.
    
    # Logic: usually the ticker is the last segment.
    return parts[-1].upper()

def main():
    parser = argparse.ArgumentParser(description="Fetch Kalshi market data from a URL.")
    parser.add_argument("url", help="The full Kalshi market URL")
    args = parser.parse_args()

    try:
        ticker = extract_ticker_from_url(args.url)
        print(f"Extracted ticker: {ticker}", file=sys.stderr)
        
        # Try fetching as a single market first
        try:
            snapshot = fetch_single_market_summary(ticker)
            print(json.dumps(snapshot.model_dump(), indent=2, default=str))
            sys.exit(0)
        except Exception:
            # Not a single market ticker, or not found.
            pass

        print(f"Ticker '{ticker}' not found as a direct market. Checking series/events...", file=sys.stderr)

        # Try fetching markets by series ticker
        resp = kalshi_service.get_markets(series_ticker=ticker, limit=10)
        markets = resp.get("markets", [])
        if markets:
            print(f"Found {len(markets)} markets for series '{ticker}':", file=sys.stderr)
            # Fetch full summaries for these
            for m in markets:
                m_ticker = m.get("ticker") or m.get("id")
                try:
                    s = fetch_single_market_summary(m_ticker)
                    print(json.dumps(s.model_dump(), indent=2, default=str))
                    print("-" * 20)
                except Exception:
                     pass
            sys.exit(0)

        # Try fetching markets by event ticker
        resp = kalshi_service.get_markets(event_ticker=ticker, limit=10)
        markets = resp.get("markets", [])
        if markets:
            print(f"Found {len(markets)} markets for event '{ticker}':", file=sys.stderr)
            for m in markets:
                m_ticker = m.get("ticker") or m.get("id")
                try:
                    s = fetch_single_market_summary(m_ticker)
                    print(json.dumps(s.model_dump(), indent=2, default=str))
                    print("-" * 20)
                except Exception:
                     pass
            sys.exit(0)
            
        print(f"No markets found for ticker/series/event: {ticker}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error fetching market: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
