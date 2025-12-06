"""
Pytest configuration and shared fixtures.
"""

import pytest
from pathlib import Path

# Project root for test data
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_market_data():
    """Sample Kalshi market data for testing."""
    return {
        "ticker": "ROTTOM-TEST-T75",
        "title": "Will Test Movie have a Rotten Tomatoes score above 75?",
        "category": "Entertainment",
        "status": "active",
        "series_ticker": "ROTTOM",
        "event_ticker": "ROTTOM-TEST",
        "volume_24h": 5000,
        "open_interest": 1200,
        "close_time": "2024-12-25T00:00:00Z",
        "yes_bid": 62,
        "yes_ask": 65,
        "no_bid": 35,
        "no_ask": 38,
    }


@pytest.fixture
def sample_event_data(sample_market_data):
    """Sample Kalshi event data for testing."""
    return {
        "event": {
            "event_ticker": "ROTTOM-TEST",
            "title": "Test Movie Rotten Tomatoes Score",
            "category": "Entertainment",
            "series_ticker": "ROTTOM",
        },
        "markets": [
            {**sample_market_data, "floor_strike": 60},
            {**sample_market_data, "ticker": "ROTTOM-TEST-T90", "floor_strike": 90, "yes_bid": 25, "yes_ask": 30},
        ],
    }


@pytest.fixture
def sample_features():
    """Sample prediction features."""
    return {
        "days_since_release": 7.0,
        "current_rating": 72.5,
        "num_reviews": 125.0,
    }


@pytest.fixture
def sample_features_partial():
    """Sample features with some missing values."""
    return {
        "current_rating": 80.0,
    }


