"""
Pydantic schemas for type-safe data structures.
Replaces Dict[str, Any] with validated models for better IDE support and runtime safety.
"""

from app.schemas.market import (
    MarketContext,
    MarketSummary,
    MarketsResponse,
    SearchResponse,
    SearchResult,
)
from app.schemas.prediction import (
    PredictionFeatures,
    PredictionResult,
    RTContext,
    SignalResponse,
)
from app.schemas.validation import (
    MarketInputParam,
    PaginationParam,
    PredictionFeaturesParam,
    SearchQueriesParam,
    SearchQueryParam,
    SignalRequestParam,
    TickerParam,
)

__all__ = [
    # Market schemas
    "MarketSummary",
    "MarketContext",
    "MarketsResponse",
    "SearchResult",
    "SearchResponse",
    # Prediction schemas
    "PredictionFeatures",
    "PredictionResult",
    "RTContext",
    "SignalResponse",
    # Validation schemas
    "TickerParam",
    "MarketInputParam",
    "SearchQueryParam",
    "SearchQueriesParam",
    "PredictionFeaturesParam",
    "SignalRequestParam",
    "PaginationParam",
]
