import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from services.gemini_integration import generate_text_async
from services.kalshi_api import (
    fetch_markets_from_kalshi,
    fetch_single_market_snapshot,
    get_orderbook_snapshot,
    search_markets,
)
from services.prediction_service import prediction_engine

app = FastAPI(title="Rotten Tomatoes Predictor")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _parse_query_terms(raw_queries: List[str]) -> List[str]:
    """Split/trim query terms from repeated params, commas, or newlines."""
    terms: List[str] = []
    for raw in raw_queries:
        if not raw:
            continue
        for candidate in raw.replace(",", "\n").split("\n"):
            cleaned = candidate.strip()
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
    return terms


@app.get("/")
async def home(request: Request):
    markets = fetch_markets_from_kalshi(limit=18, depth=5)
    return templates.TemplateResponse("index.html", {"request": request, "markets": markets})


@app.get("/ask-ai")
async def ask_ai(prompt: str):
    response = await generate_text_async(prompt)
    return {"response": response}


@app.get("/search")
async def search(request: Request, q: List[str] = Query(default=[]), depth: int = 3):
    """
    GUI search for markets (multi-term; comma or newline separated).
    """
    queries = _parse_query_terms(q)
    results_map = search_markets(queries, depth=depth) if queries else {}
    results = [{"query": query, "market": results_map.get(query)} for query in queries]
    prefill_text = "\n".join(queries)
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "queries": queries,
            "results": results,
            "prefill_text": prefill_text,
            "depth": depth,
        },
    )


@app.get("/api/markets")
async def api_markets(limit: int = 20, depth: int = 5):
    """Standardized markets endpoint with enriched orderbook details."""
    markets = fetch_markets_from_kalshi(limit=limit, depth=depth)
    return JSONResponse({"markets": markets})


@app.get("/api/markets/{ticker}")
async def api_market_detail(ticker: str, depth: int = 5):
    """Get a single market snapshot (summary + orderbook)."""
    snapshot = fetch_single_market_snapshot(ticker, depth=depth)
    if not snapshot:
        return JSONResponse({"detail": "Market not found"}, status_code=404)
    return JSONResponse(snapshot)


@app.get("/api/markets/{ticker}/orderbook")
async def api_market_orderbook(ticker: str, depth: int = 5):
    """Orderbook-only view for consumers that need raw depth."""
    snapshot = get_orderbook_snapshot(ticker, depth=depth)
    return JSONResponse(snapshot)


@app.get("/api/search")
async def api_search(q: List[str] = Query(default=[]), depth: int = 3, limit: int = 150):
    """
    API search endpoint mirroring the CLI search tool.
    """
    queries = _parse_query_terms(q)
    results = search_markets(queries, depth=depth, search_limit=limit) if queries else {}
    return JSONResponse({"queries": queries, "results": results})


@app.get("/api/markets/{ticker}/signals")
async def api_market_signals(
    ticker: str,
    days_since_release: float | None = Query(default=None, description="Days since theatrical release"),
    current_rating: float | None = Query(default=None, description="Current Rotten Tomatoes rating"),
    num_reviews: float | None = Query(default=None, description="Number of RT critic reviews"),
    depth: int = 5,
):
    """
    Serve model-driven signals plus the Rotten Tomatoes + orderbook context for explainability.
    """
    rt_features = {
        "days_since_release": days_since_release,
        "current_rating": current_rating,
        "num_reviews": num_reviews,
    }

    market_snapshot = fetch_single_market_snapshot(ticker, depth=depth)
    payload = prediction_engine.generate_signal_payload(ticker, rt_features, market_snapshot)
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
