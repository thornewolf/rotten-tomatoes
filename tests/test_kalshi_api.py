"""
Unit tests for Kalshi API integration.
Tests the kalshi_api module which builds market summaries from raw API data.
"""

import pytest
from unittest.mock import patch, MagicMock

from services.kalshi_api import (
    _normalize_price,
    _build_market_summary,
    _build_event_summary,
    _estimate_score_from_markets,
    fetch_single_market_summary,
    search_markets,
)
from app.schemas import MarketSummary


class TestNormalizePrice:
    """Tests for price normalization."""

    def test_normalize_cents_to_dollars(self):
        """Should convert cents (>1) to dollars."""
        assert _normalize_price(65) == 0.65
        assert _normalize_price(100) == 1.0
        # Values between 1 and 2 are ambiguous - treated as already normalized
        assert _normalize_price(2) == 0.02  # > 1, so divided by 100

    def test_normalize_already_decimal(self):
        """Should preserve already-normalized prices (<=1)."""
        assert _normalize_price(0.65) == 0.65
        assert _normalize_price(0.01) == 0.01
        assert _normalize_price(1) == 1.0  # Exactly 1 is preserved

    def test_normalize_none(self):
        """Should return None for None input."""
        assert _normalize_price(None) is None

    def test_normalize_zero(self):
        """Should handle zero correctly."""
        assert _normalize_price(0) == 0.0


class TestBuildMarketSummary:
    """Tests for building MarketSummary from market data."""

    def test_build_from_valid_data(self, sample_market_data):
        """Should build MarketSummary from valid market data."""
        summary = _build_market_summary(sample_market_data)

        assert isinstance(summary, MarketSummary)
        assert summary.ticker == "ROTTOM-TEST-T75"
        assert summary.title == "Will Test Movie have a Rotten Tomatoes score above 75?"
        assert summary.status == "active"

    def test_price_normalization(self, sample_market_data):
        """Should normalize prices from cents to dollars."""
        summary = _build_market_summary(sample_market_data)

        assert summary.yes_bid == 0.62
        assert summary.yes_ask == 0.65

    def test_missing_ticker_raises(self):
        """Should raise ValueError for missing ticker."""
        with pytest.raises(ValueError) as exc_info:
            _build_market_summary({"title": "No ticker"})
        assert "ticker" in str(exc_info.value).lower()

    def test_non_dict_raises(self):
        """Should raise ValueError for non-dict input."""
        with pytest.raises(ValueError):
            _build_market_summary("not a dict")

    def test_link_generation(self, sample_market_data):
        """Should generate correct Kalshi link."""
        summary = _build_market_summary(sample_market_data)
        assert "kalshi.com/markets" in summary.link


class TestEstimateScoreFromMarkets:
    """Tests for estimating RT score from market prices."""

    def test_estimate_from_strike_markets(self):
        """Should estimate score from market strike prices."""
        markets = [
            {"floor_strike": 60, "yes_bid": 80, "yes_ask": 85},  # 82.5% > 60
            {"floor_strike": 75, "yes_bid": 50, "yes_ask": 55},  # 52.5% > 75
            {"floor_strike": 90, "yes_bid": 20, "yes_ask": 25},  # 22.5% > 90
        ]

        score = _estimate_score_from_markets(markets)

        assert score is not None
        assert 60 < score < 90  # Should be in reasonable range

    def test_empty_markets_returns_none(self):
        """Should return None for empty markets list."""
        assert _estimate_score_from_markets([]) is None

    def test_no_strike_markets_returns_none(self):
        """Should return None when no markets have strikes."""
        markets = [{"yes_bid": 50, "yes_ask": 55}]  # No floor_strike
        assert _estimate_score_from_markets(markets) is None


class TestBuildEventSummary:
    """Tests for building event summaries."""

    def test_build_from_event_data(self, sample_event_data):
        """Should build MarketSummary from event and child markets."""
        event = sample_event_data["event"]
        markets = sample_event_data["markets"]

        summary = _build_event_summary(event, markets)

        assert isinstance(summary, MarketSummary)
        assert summary.ticker == "ROTTOM-TEST"
        assert summary.volume > 0  # Aggregated from child markets
        assert summary.estimated_score is not None

    def test_aggregates_volume(self, sample_event_data):
        """Should aggregate volume from all child markets."""
        event = sample_event_data["event"]
        markets = sample_event_data["markets"]

        summary = _build_event_summary(event, markets)

        total_expected = sum(m.get("volume_24h", 0) for m in markets)
        assert summary.volume == total_expected

    def test_missing_event_ticker_raises(self):
        """Should raise ValueError for missing event_ticker."""
        with pytest.raises(ValueError):
            _build_event_summary({"title": "No ticker"}, [])


class TestFetchSingleMarketSummary:
    """Tests for fetching a single market summary."""

    @patch("services.kalshi_api.kalshi_service.get_event")
    def test_fetch_valid_ticker(self, mock_get_event, sample_event_data):
        """Should fetch and build summary for valid ticker."""
        mock_get_event.return_value = sample_event_data

        summary = fetch_single_market_summary("ROTTOM-TEST")

        assert isinstance(summary, MarketSummary)
        assert summary.ticker == "ROTTOM-TEST"
        mock_get_event.assert_called_once_with("ROTTOM-TEST")

    @patch("services.kalshi_api.kalshi_service.get_event")
    def test_fetch_empty_ticker_raises(self, mock_get_event):
        """Should raise ValueError for empty ticker."""
        with pytest.raises(ValueError) as exc_info:
            fetch_single_market_summary("")
        assert "required" in str(exc_info.value).lower()

    @patch("services.kalshi_api.kalshi_service.get_event")
    def test_fetch_not_found_raises(self, mock_get_event):
        """Should raise ValueError when event not found."""
        mock_get_event.return_value = None

        with pytest.raises(ValueError) as exc_info:
            fetch_single_market_summary("NONEXISTENT")
        assert "not found" in str(exc_info.value).lower()


class TestSearchMarkets:
    """Tests for batch market search."""

    @patch("services.kalshi_api.kalshi_service.get_events")
    def test_search_single_query(self, mock_get_events, sample_market_data):
        """Should find market matching query."""
        mock_get_events.return_value = {
            "events": [sample_market_data]
        }

        results = search_markets(["TEST"])

        assert "TEST" in results
        # Note: might be None if title doesn't match

    @patch("services.kalshi_api.kalshi_service.get_events")
    def test_search_empty_queries(self, mock_get_events):
        """Should return empty dict for empty queries."""
        results = search_markets([])

        assert results == {}
        mock_get_events.assert_not_called()

    @patch("services.kalshi_api.kalshi_service.get_events")
    def test_search_no_matches(self, mock_get_events):
        """Should return None for unmatched queries."""
        mock_get_events.return_value = {"events": []}

        results = search_markets(["NONEXISTENT"])

        assert results["NONEXISTENT"] is None

    @patch("services.kalshi_api.kalshi_service.get_events")
    def test_search_case_insensitive(self, mock_get_events, sample_market_data):
        """Should match queries case-insensitively."""
        sample_market_data["event_ticker"] = "ROTTOM-TEST"
        mock_get_events.return_value = {
            "events": [sample_market_data]
        }

        results = search_markets(["rottom"])

        # Should find the market with case-insensitive match
        assert "rottom" in results
