"""
Input validation schemas for API parameters.
Provides strict validation for all user inputs to prevent invalid data and security issues.
"""

import re
from typing import Annotated, List

from pydantic import BaseModel, Field, field_validator


# Regex patterns for validation
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9\-]{0,49}$")
SEARCH_QUERY_PATTERN = re.compile(r"^[\w\s\-\.\,\'\"\!\?\&\(\)]{1,200}$", re.UNICODE)


class TickerParam(BaseModel):
    """Validated ticker parameter."""

    ticker: str = Field(..., min_length=1, max_length=50, description="Market ticker symbol")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate ticker format - alphanumeric and hyphens only, uppercase."""
        v = v.strip().upper()
        if not v:
            raise ValueError("Ticker cannot be empty")
        if not TICKER_PATTERN.match(v):
            raise ValueError(
                f"Invalid ticker format: '{v}'. "
                "Must be 1-50 characters, alphanumeric and hyphens only, starting with alphanumeric."
            )
        return v


class MarketInputParam(BaseModel):
    """Validated market input (ticker or URL)."""

    market_input: str = Field(..., min_length=1, max_length=500, description="Ticker or Kalshi URL")

    @field_validator("market_input")
    @classmethod
    def validate_market_input(cls, v: str) -> str:
        """Validate market input - either a URL or ticker."""
        v = v.strip()
        if not v:
            raise ValueError("Market input cannot be empty")

        # Allow URLs
        if v.startswith(("http://", "https://")):
            if "kalshi.com" not in v.lower():
                raise ValueError("Only Kalshi URLs are accepted")
            if len(v) > 500:
                raise ValueError("URL too long")
            return v

        # Otherwise validate as ticker
        v_upper = v.upper()
        if not TICKER_PATTERN.match(v_upper):
            raise ValueError(
                f"Invalid ticker format: '{v}'. "
                "Must be alphanumeric and hyphens only."
            )
        return v_upper


class SearchQueryParam(BaseModel):
    """Validated search query parameter."""

    query: str = Field(..., min_length=1, max_length=200, description="Search query")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate search query - prevent injection attacks."""
        v = v.strip()
        if not v:
            raise ValueError("Search query cannot be empty")
        if len(v) > 200:
            raise ValueError("Search query too long (max 200 characters)")
        # Allow common characters but prevent script injection
        if "<" in v or ">" in v or "javascript:" in v.lower():
            raise ValueError("Invalid characters in search query")
        return v


class SearchQueriesParam(BaseModel):
    """Validated list of search queries."""

    queries: List[str] = Field(default_factory=list, max_length=50, description="List of search queries")

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, v: List[str]) -> List[str]:
        """Validate all search queries."""
        if len(v) > 50:
            raise ValueError("Too many search queries (max 50)")

        validated = []
        for q in v:
            q = q.strip()
            if not q:
                continue
            if len(q) > 200:
                raise ValueError(f"Query too long: '{q[:50]}...'")
            if "<" in q or ">" in q or "javascript:" in q.lower():
                raise ValueError(f"Invalid characters in query: '{q[:50]}'")
            validated.append(q)
        return validated


class PredictionFeaturesParam(BaseModel):
    """Validated prediction feature parameters."""

    days_since_release: float | None = Field(
        default=None,
        ge=0,
        le=365,
        description="Days since theatrical release (0-365)"
    )
    current_rating: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Current Rotten Tomatoes rating (0-100)"
    )
    num_reviews: float | None = Field(
        default=None,
        ge=0,
        le=1000,
        description="Number of RT critic reviews (0-1000)"
    )

    @field_validator("days_since_release", "current_rating", "num_reviews", mode="before")
    @classmethod
    def validate_numeric(cls, v):
        """Convert and validate numeric values."""
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {v}")


class SignalRequestParam(BaseModel):
    """Complete validated signal request parameters."""

    ticker: str = Field(..., min_length=1, max_length=50)
    days_since_release: float | None = Field(default=None, ge=0, le=365)
    current_rating: float | None = Field(default=None, ge=0, le=100)
    num_reviews: float | None = Field(default=None, ge=0, le=1000)
    model: str | None = Field(default=None, max_length=50)
    depth: int = Field(default=5, ge=1, le=20)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not TICKER_PATTERN.match(v):
            raise ValueError(f"Invalid ticker format: '{v}'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        if not re.match(r"^[a-z0-9_\-]{1,50}$", v):
            raise ValueError(f"Invalid model name: '{v}'")
        return v


class PaginationParam(BaseModel):
    """Validated pagination parameters."""

    limit: int = Field(default=20, ge=1, le=100, description="Number of results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


# Type aliases for FastAPI dependency injection
ValidatedTicker = Annotated[str, Field(min_length=1, max_length=50, pattern=r"^[A-Z0-9][A-Z0-9\-]{0,49}$")]
ValidatedSearchQuery = Annotated[str, Field(min_length=1, max_length=200)]
