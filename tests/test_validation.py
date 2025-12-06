"""
Unit tests for input validation schemas.
"""

import pytest
from pydantic import ValidationError

from app.schemas.validation import (
    TickerParam,
    MarketInputParam,
    SearchQueryParam,
    SearchQueriesParam,
    PredictionFeaturesParam,
    SignalRequestParam,
    PaginationParam,
)


class TestTickerParam:
    """Tests for ticker validation."""

    def test_valid_ticker(self):
        """Should accept valid ticker formats."""
        valid_tickers = [
            "AAPL",
            "ROTTOM-24DEC",
            "TEST123",
            "A",
            "ABC-DEF-GHI",
        ]
        for ticker in valid_tickers:
            result = TickerParam(ticker=ticker)
            assert result.ticker == ticker.upper()

    def test_ticker_uppercase_conversion(self):
        """Should convert ticker to uppercase."""
        result = TickerParam(ticker="aapl")
        assert result.ticker == "AAPL"

    def test_ticker_strips_whitespace(self):
        """Should strip whitespace."""
        result = TickerParam(ticker="  AAPL  ")
        assert result.ticker == "AAPL"

    def test_invalid_ticker_empty(self):
        """Should reject empty ticker."""
        with pytest.raises(ValidationError):
            TickerParam(ticker="")

    def test_invalid_ticker_whitespace_only(self):
        """Should reject whitespace-only ticker."""
        with pytest.raises(ValidationError):
            TickerParam(ticker="   ")

    def test_invalid_ticker_special_chars(self):
        """Should reject tickers with invalid characters."""
        invalid_tickers = [
            "AAP$L",
            "TEST@123",
            "TICK ER",
            "test/abc",
            "-STARTS-DASH",
        ]
        for ticker in invalid_tickers:
            with pytest.raises(ValidationError):
                TickerParam(ticker=ticker)


class TestMarketInputParam:
    """Tests for market input (ticker or URL) validation."""

    def test_valid_ticker_input(self):
        """Should accept valid ticker."""
        result = MarketInputParam(market_input="ROTTOM-24DEC")
        assert result.market_input == "ROTTOM-24DEC"

    def test_valid_kalshi_url(self):
        """Should accept Kalshi URLs."""
        urls = [
            "https://kalshi.com/markets/ROTTOM",
            "https://www.kalshi.com/events/TEST",
            "http://kalshi.com/markets/ABC",
        ]
        for url in urls:
            result = MarketInputParam(market_input=url)
            assert result.market_input == url

    def test_invalid_non_kalshi_url(self):
        """Should reject non-Kalshi URLs."""
        with pytest.raises(ValidationError) as exc_info:
            MarketInputParam(market_input="https://example.com/markets/TEST")
        assert "kalshi" in str(exc_info.value).lower()

    def test_invalid_empty(self):
        """Should reject empty input."""
        with pytest.raises(ValidationError):
            MarketInputParam(market_input="")


class TestSearchQueryParam:
    """Tests for search query validation."""

    def test_valid_query(self):
        """Should accept valid search queries."""
        queries = [
            "Rotten Tomatoes",
            "Test Movie 2024",
            "What's happening?",
        ]
        for query in queries:
            result = SearchQueryParam(query=query)
            assert result.query == query

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        result = SearchQueryParam(query="  search term  ")
        assert result.query == "search term"

    def test_rejects_empty(self):
        """Should reject empty query."""
        with pytest.raises(ValidationError):
            SearchQueryParam(query="")

    def test_rejects_script_injection(self):
        """Should reject potential XSS attempts."""
        dangerous = [
            "<script>alert(1)</script>",
            "javascript:void(0)",
            "<img onerror=alert(1)>",
        ]
        for query in dangerous:
            with pytest.raises(ValidationError):
                SearchQueryParam(query=query)


class TestSearchQueriesParam:
    """Tests for batch search queries validation."""

    def test_valid_queries_list(self):
        """Should accept valid list of queries."""
        result = SearchQueriesParam(queries=["query1", "query2", "query3"])
        assert result.queries == ["query1", "query2", "query3"]

    def test_filters_empty_queries(self):
        """Should filter out empty queries."""
        result = SearchQueriesParam(queries=["valid", "", "  ", "also valid"])
        assert result.queries == ["valid", "also valid"]

    def test_rejects_too_many_queries(self):
        """Should reject more than 50 queries."""
        too_many = [f"query{i}" for i in range(60)]
        with pytest.raises(ValidationError):
            SearchQueriesParam(queries=too_many)

    def test_rejects_long_query(self):
        """Should reject queries longer than 200 chars."""
        long_query = "x" * 250
        with pytest.raises(ValidationError):
            SearchQueriesParam(queries=[long_query])


class TestPredictionFeaturesParam:
    """Tests for prediction feature validation."""

    def test_valid_features(self):
        """Should accept valid feature values."""
        result = PredictionFeaturesParam(
            days_since_release=7.0,
            current_rating=75.5,
            num_reviews=100.0,
        )
        assert result.days_since_release == 7.0
        assert result.current_rating == 75.5
        assert result.num_reviews == 100.0

    def test_optional_features(self):
        """All features should be optional."""
        result = PredictionFeaturesParam()
        assert result.days_since_release is None
        assert result.current_rating is None
        assert result.num_reviews is None

    def test_rejects_negative_values(self):
        """Should reject negative values."""
        with pytest.raises(ValidationError):
            PredictionFeaturesParam(days_since_release=-1.0)

        with pytest.raises(ValidationError):
            PredictionFeaturesParam(num_reviews=-10.0)

    def test_rejects_rating_over_100(self):
        """Should reject rating > 100."""
        with pytest.raises(ValidationError):
            PredictionFeaturesParam(current_rating=105.0)

    def test_rejects_days_over_365(self):
        """Should reject days > 365."""
        with pytest.raises(ValidationError):
            PredictionFeaturesParam(days_since_release=400.0)


class TestSignalRequestParam:
    """Tests for complete signal request validation."""

    def test_valid_request(self):
        """Should accept valid complete request."""
        result = SignalRequestParam(
            ticker="ROTTOM-24DEC",
            days_since_release=7.0,
            current_rating=75.0,
            num_reviews=100.0,
            model="default",
            depth=5,
        )
        assert result.ticker == "ROTTOM-24DEC"
        assert result.model == "default"

    def test_model_lowercase_conversion(self):
        """Should convert model name to lowercase."""
        result = SignalRequestParam(ticker="TEST", model="DEFAULT")
        assert result.model == "default"

    def test_rejects_invalid_model_name(self):
        """Should reject invalid model names."""
        with pytest.raises(ValidationError):
            SignalRequestParam(ticker="TEST", model="invalid model!")


class TestPaginationParam:
    """Tests for pagination parameter validation."""

    def test_default_values(self):
        """Should have sensible defaults."""
        result = PaginationParam()
        assert result.limit == 20
        assert result.offset == 0

    def test_custom_values(self):
        """Should accept custom values within bounds."""
        result = PaginationParam(limit=50, offset=100)
        assert result.limit == 50
        assert result.offset == 100

    def test_rejects_limit_over_100(self):
        """Should reject limit > 100."""
        with pytest.raises(ValidationError):
            PaginationParam(limit=150)

    def test_rejects_negative_offset(self):
        """Should reject negative offset."""
        with pytest.raises(ValidationError):
            PaginationParam(offset=-10)
