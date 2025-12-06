import logging
from pathlib import Path
from typing import Annotated, List
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Form, Path as PathParam, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from app.schemas.validation import MarketInputParam, SearchQueriesParam, TickerParam
from app.utils import parse_query_terms
from core.model_registry import get_model_registry
from core.security import KalshiRateLimitError, KalshiServerError
from services.gemini_integration import generate_text_async
from services.kalshi_api import (
    fetch_single_market_summary,
    search_markets,
)
from services.prediction_service import get_prediction_engine

app = FastAPI(title="Rotten Tomatoes Predictor")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TRACKED_MARKETS_FILE = PROJECT_ROOT / "configs" / "tracked_markets.txt"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def extract_ticker_from_input(raw: str) -> str | None:
    """Extract a market ticker from a URL or return the ticker directly."""
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return None

    ticker = None
    # Check if it's a URL
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        # Kalshi URLs: https://kalshi.com/markets/TICKER or .../events/TICKER
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 2 and path_parts[0] in ("markets", "events"):
            ticker = path_parts[1]
        # Fallback: last path segment
        elif path_parts:
            ticker = path_parts[-1]
    else:
        # Assume it's already a ticker
        ticker = raw

    # Kalshi tickers/events must be uppercase
    return ticker.upper() if ticker else None


def load_tracked_tickers() -> List[str]:
    """Load tracked market tickers from file."""
    if not TRACKED_MARKETS_FILE.exists():
        return []
    tickers = []
    for line in TRACKED_MARKETS_FILE.read_text().splitlines():
        ticker = extract_ticker_from_input(line)
        if ticker:
            tickers.append(ticker)
    return tickers


def save_tracked_ticker(ticker: str) -> None:
    """Append a ticker to the tracked markets file."""
    TRACKED_MARKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = load_tracked_tickers()
    if ticker not in existing:
        with TRACKED_MARKETS_FILE.open("a") as f:
            f.write(f"{ticker}\n")

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
async def home(request: Request):
    error = None
    markets = []
    tracked_tickers = load_tracked_tickers()

    for ticker in tracked_tickers:
        try:
            market = fetch_single_market_summary(ticker)
            markets.append(market)
        except (requests.RequestException, KalshiRateLimitError, KalshiServerError) as exc:
            logger.warning("API error fetching tracked market %s: %s", ticker, exc)
        except ValueError as exc:
            logger.warning("Invalid data for market %s: %s", ticker, exc)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "markets": markets, "error": error, "tracked_count": len(tracked_tickers)},
    )


@app.post("/track")
async def add_tracked_market(market_input: str = Form(..., max_length=500)):
    """Add a market to the tracked list by ticker or URL."""
    # Validate input format
    try:
        validated = MarketInputParam(market_input=market_input)
        ticker = extract_ticker_from_input(validated.market_input)
    except ValidationError as exc:
        logger.warning("Invalid market input: %s", exc)
        return RedirectResponse(url="/?error=invalid_input", status_code=303)

    if not ticker:
        return RedirectResponse(url="/?error=invalid_input", status_code=303)

    # Validate the ticker exists by fetching it
    try:
        fetch_single_market_summary(ticker)
    except (requests.RequestException, KalshiRateLimitError, KalshiServerError) as exc:
        logger.warning("API error validating market %s: %s", ticker, exc)
        return RedirectResponse(url=f"/?error=api_error&ticker={ticker}", status_code=303)
    except ValueError as exc:
        logger.warning("Market not found or invalid %s: %s", ticker, exc)
        return RedirectResponse(url=f"/?error=market_not_found&ticker={ticker}", status_code=303)

    save_tracked_ticker(ticker)
    return RedirectResponse(url="/", status_code=303)


@app.post("/untrack/{ticker}")
async def remove_tracked_market(
    ticker: Annotated[str, PathParam(min_length=1, max_length=50, pattern=r"^[A-Z0-9][A-Z0-9\-]{0,49}$")]
):
    """Remove a market from the tracked list."""
    if not TRACKED_MARKETS_FILE.exists():
        return RedirectResponse(url="/", status_code=303)

    lines = TRACKED_MARKETS_FILE.read_text().splitlines()
    new_lines = []
    for line in lines:
        extracted = extract_ticker_from_input(line)
        if extracted != ticker:
            new_lines.append(line)
    TRACKED_MARKETS_FILE.write_text("\n".join(new_lines) + "\n" if new_lines else "")
    return RedirectResponse(url="/", status_code=303)


@app.get("/ask-ai")
async def ask_ai(prompt: str):
    response = await generate_text_async(prompt)
    return {"response": response}


@app.get("/search")
async def search(request: Request, q: List[str] = Query(default=[], max_length=50)):
    """
    GUI search for markets (multi-term; comma or newline separated).
    """
    # Validate and sanitize queries
    try:
        validated = SearchQueriesParam(queries=q)
        queries = parse_query_terms(validated.queries)
    except ValidationError as exc:
        logger.warning("Invalid search queries: %s", exc)
        queries = []
    results_map = {}
    search_error = None
    if queries:
        try:
            results_map = search_markets(queries)
        except (requests.RequestException, KalshiRateLimitError, KalshiServerError) as exc:
            logger.warning("API error during search: %s", exc)
            search_error = f"API error: {exc}"
        except ValueError as exc:
            logger.warning("Invalid search response: %s", exc)
            search_error = str(exc)
    results = [{"query": query, "market": results_map.get(query)} for query in queries]
    prefill_text = "\n".join(queries)
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "queries": queries,
            "results": results,
            "prefill_text": prefill_text,
            "search_error": search_error,
        },
    )


@app.get("/api/markets")
async def api_markets(
    limit: Annotated[int, Query(ge=1, le=100, description="Max results")] = 20,
    depth: Annotated[int, Query(ge=1, le=20, description="Search depth")] = 5
):
    markets = fetch_market_summaries(limit=limit)
    return JSONResponse({"markets": [m.model_dump() for m in markets]})


@app.get("/api/markets/{ticker}")
async def api_market_detail(
    ticker: Annotated[str, PathParam(min_length=1, max_length=50)],
    depth: Annotated[int, Query(ge=1, le=20)] = 5
):
    """Get a single market summary."""
    # Validate ticker format
    try:
        validated = TickerParam(ticker=ticker)
        ticker = validated.ticker
    except ValidationError as exc:
        return JSONResponse({"detail": "Invalid ticker format", "errors": exc.errors()}, status_code=400)

    try:
        summary = fetch_single_market_summary(ticker)
    except (requests.RequestException, KalshiRateLimitError, KalshiServerError) as exc:
        logger.warning("API error fetching market %s: %s", ticker, exc)
        return JSONResponse({"detail": "API error", "error": str(exc)}, status_code=503)
    except ValueError as exc:
        logger.warning("Market not found %s: %s", ticker, exc)
        return JSONResponse({"detail": "Market not found"}, status_code=404)
    return JSONResponse(summary.model_dump())


@app.get("/api/search")
async def api_search(
    q: List[str] = Query(default=[], max_length=50),
    depth: Annotated[int, Query(ge=1, le=20)] = 3,
    limit: Annotated[int, Query(ge=1, le=500)] = 150
):
    """
    API search endpoint mirroring the CLI search tool.
    """
    # Validate queries
    try:
        validated = SearchQueriesParam(queries=q)
        queries = parse_query_terms(validated.queries)
    except ValidationError as exc:
        return JSONResponse({"detail": "Invalid search queries", "errors": exc.errors()}, status_code=400)

    results = search_markets(queries, depth=depth, search_limit=limit) if queries else {}
    # Convert MarketSummary objects to dicts for JSON serialization
    serialized_results = {
        query: market.model_dump() if market else None for query, market in results.items()
    }
    return JSONResponse({"queries": queries, "results": serialized_results})


@app.get("/markets/{ticker}")
async def market_detail(
    request: Request,
    ticker: Annotated[str, PathParam(min_length=1, max_length=50)]
):
    """HTML detail page for a single market summary."""
    # Validate ticker
    try:
        validated = TickerParam(ticker=ticker)
        ticker = validated.ticker
    except ValidationError:
        return templates.TemplateResponse(
            "market_detail.html",
            {"request": request, "market": None, "error": "Invalid ticker format", "ticker": ticker},
        )

    error = None
    snapshot = None
    try:
        snapshot = fetch_single_market_summary(ticker)
    except (requests.RequestException, KalshiRateLimitError, KalshiServerError) as exc:
        error = f"API error: {exc}"
        logger.warning("API error fetching market detail for %s: %s", ticker, exc)
    except ValueError as exc:
        error = str(exc)
        logger.warning("Market not found %s: %s", ticker, exc)
    if not snapshot and not error:
        error = "Market not found"
    return templates.TemplateResponse(
        "market_detail.html",
        {"request": request, "market": snapshot, "error": error, "ticker": ticker},
    )


def get_available_models() -> list[str]:
    """Get list of available model names from registry."""
    registry = get_model_registry()
    return registry.list_models()


@app.get("/api/models")
async def api_list_models():
    """List available prediction models."""
    registry = get_model_registry()
    models = []
    for name in registry.list_models():
        config = registry.get(name)
        if config:
            models.append({
                "name": config.name,
                "description": config.description,
                "is_default": config.is_default,
                "available": config.exists(),
            })
    return JSONResponse({"models": models})


@app.get("/api/markets/{ticker}/signals")
async def api_market_signals(
    ticker: Annotated[str, PathParam(min_length=1, max_length=50)],
    days_since_release: Annotated[float | None, Query(ge=0, le=365, description="Days since theatrical release")] = None,
    current_rating: Annotated[float | None, Query(ge=0, le=100, description="Current Rotten Tomatoes rating")] = None,
    num_reviews: Annotated[float | None, Query(ge=0, le=10000, description="Number of RT critic reviews")] = None,
    model: Annotated[str | None, Query(max_length=50, description="Model name from registry")] = None,
    depth: Annotated[int, Query(ge=1, le=20)] = 5,
):
    """
    Serve model-driven signals plus Rotten Tomatoes + market summary context.
    """
    from services.prediction_service import PredictionEngine

    # Validate ticker
    try:
        validated = TickerParam(ticker=ticker)
        ticker = validated.ticker
    except ValidationError as exc:
        return JSONResponse({"detail": "Invalid ticker format", "errors": exc.errors()}, status_code=400)

    rt_features = {
        "days_since_release": days_since_release,
        "current_rating": current_rating,
        "num_reviews": num_reviews,
    }

    market_snapshot = fetch_single_market_summary(ticker)

    # Select model from registry
    registry = get_model_registry()
    if model and registry.get(model):
        model_config = registry.get(model)
    else:
        model_config = registry.get_default()

    model_path = model_config.path if model_config else None
    engine = PredictionEngine(model_path=model_path) if model_path else PredictionEngine()

    payload = engine.generate_signal_payload(ticker, rt_features, market_snapshot)
    return JSONResponse(payload.model_dump())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
