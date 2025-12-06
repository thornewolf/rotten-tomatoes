"""
Pydantic models for Kalshi market data structures.
Provides type safety and validation for market-related API responses.
"""

from pydantic import BaseModel, Field


class MarketSummary(BaseModel):
    """Summary of a Kalshi prediction market."""

    ticker: str = Field(..., description="Market ticker symbol")
    title: str | None = Field(None, description="Human-readable market title")
    category: str | None = Field(None, description="Market category")
    status: str | None = Field(None, description="Market status (open, closed, settled)")
    series_ticker: str | None = Field(None, description="Series ticker this market belongs to")
    event_ticker: str | None = Field(None, description="Event ticker this market belongs to")
    volume: float | None = Field(None, description="24-hour trading volume")
    open_interest: float | None = Field(None, description="Current open interest")
    close_time: str | None = Field(None, description="Market close timestamp (ISO 8601)")
    yes_bid: float | None = Field(None, ge=0.0, le=1.0, description="Current highest YES bid price (0-1)")
    yes_ask: float | None = Field(None, ge=0.0, le=1.0, description="Current lowest YES ask price (0-1)")
    no_bid: float | None = Field(None, ge=0.0, le=1.0, description="Current highest NO bid price (0-1)")
    no_ask: float | None = Field(None, ge=0.0, le=1.0, description="Current lowest NO ask price (0-1)")
    link: str = Field(..., description="URL to market page on Kalshi")
    estimated_score: float | None = Field(None, description="Market-implied point estimate for RT score")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "ROTTOM-23DEC-T75",
                "title": "Will Tron: Legacy have a Rotten Tomatoes score above 75?",
                "category": "Entertainment",
                "status": "open",
                "series_ticker": "ROTTOM",
                "event_ticker": "ROTTOM-23DEC",
                "volume": 12500.0,
                "open_interest": 3200.0,
                "close_time": "2023-12-24T00:00:00Z",
                "yes_bid": 0.62,
                "yes_ask": 0.65,
                "no_bid": 0.35,
                "no_ask": 0.38,
                "link": "https://kalshi.com/markets/ROTTOM",
            }
        }


class MarketContext(BaseModel):
    """Market context included in signal responses."""

    market_yes_mid_price: float | None = Field(None, description="Mid price between yes_bid and yes_ask")
    market_summary: dict = Field(default_factory=dict, description="Full market summary data")


class MarketsResponse(BaseModel):
    """Response from /api/markets endpoint."""

    markets: list[MarketSummary] = Field(default_factory=list, description="List of market summaries")


class SearchResult(BaseModel):
    """Single search result mapping query to market or None."""

    query: str = Field(..., description="The search query")
    market: MarketSummary | None = Field(None, description="Matched market or None if not found")


class SearchResponse(BaseModel):
    """Response from /api/search endpoint."""

    queries: list[str] = Field(default_factory=list, description="List of search queries")
    results: dict[str, MarketSummary | None] = Field(
        default_factory=dict, description="Map of query to market summary or None"
    )
