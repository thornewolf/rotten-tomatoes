"""
Backtesting framework for RT prediction market betting models.

Evaluates model performance using historical review data, price history,
and Kelly criterion betting with strict data leakage prevention.

Usage:
    from services.backtest import (
        BacktestConfig,
        run_backtest,
        run_backtest_from_dataframes,
        load_price_history,
        load_review_history,
    )
    from datetime import date

    # Run a backtest from CSV files
    result = run_backtest(
        movie_name="Moana 2",
        review_csv="data/moana2_reviews.csv",
        price_csv="data/moana2_prices.csv",
        release_date=date(2025, 11, 27),
        final_score=63.0,  # Actual final RT score
    )

    print(result.summary())

Data Formats:
    Price History CSV:
        timestamp,Above 10,Above 15,Above 20,...,Above 90
        2025-11-13T00:00:00Z,,,38.12,18.54,...

        - Prices in cents (0-100), converted internally to probabilities (0-1)
        - NaN/empty values indicate market not available at that time

    Review History CSV:
        Review Order,Critic Name,Date Reviewed,Review Sentiment
        1,Critic A,7-Oct,Positive
        2,Critic B,7-Oct,Negative

        - Date format configurable (default: "%d-%b")
        - Sentiment values: "positive" or "negative" (case-insensitive)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from services.predictors import BasePredictor, MLPredictor, DummyPredictor

logger = logging.getLogger(__name__)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BetRecord",
    "ReviewSnapshot",
    "Backtester",
    "run_backtest",
    "run_backtest_from_dataframes",
    "load_price_history",
    "load_review_history",
    "compute_rating_at_time",
    "kelly_bet_size",
    "parse_market_threshold",
]


@dataclass
class BacktestConfig:
    """Configuration for backtest parameters."""

    # Kelly fraction (1.0 = full Kelly, 0.5 = half Kelly recommended)
    kelly_fraction: float = 0.5

    # Starting bankroll for simulation
    initial_bankroll: float = 1000.0

    # Minimum edge required to place a bet (e.g., 0.05 = 5% edge)
    min_edge_threshold: float = 0.02

    # Maximum fraction of bankroll per single bet
    max_bet_fraction: float = 0.25

    # Minimum bet size (in dollars)
    min_bet_size: float = 1.0

    # Transaction cost per trade (as fraction of bet, e.g., 0.01 = 1%)
    transaction_cost: float = 0.0


@dataclass
class BetRecord:
    """Record of a single bet placed during backtest."""

    timestamp: datetime
    market: str  # e.g., "Above 60"
    direction: str  # "YES" or "NO"
    model_prob: float  # Model's predicted probability
    market_price: float  # Market price at time of bet (0-1 scale)
    edge: float  # model_prob - market_price for YES, or (1-model_prob) - (1-market_price) for NO
    kelly_size: float  # Recommended Kelly bet size (fraction)
    actual_bet_size: float  # Actual bet size after constraints
    bet_amount: float  # Dollar amount bet
    outcome: Optional[bool] = None  # True if bet won, False if lost, None if pending
    pnl: Optional[float] = None  # Profit/loss from this bet


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    movie_name: str
    release_date: date
    final_score: float

    # Performance metrics
    total_pnl: float = 0.0
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0

    # Bankroll tracking
    initial_bankroll: float = 1000.0
    final_bankroll: float = 1000.0
    max_drawdown: float = 0.0
    peak_bankroll: float = 1000.0

    # Individual bet records
    bets: list[BetRecord] = field(default_factory=list)

    # Daily bankroll history
    bankroll_history: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.winning_bets / self.total_bets

    @property
    def roi(self) -> float:
        if self.initial_bankroll == 0:
            return 0.0
        return (self.final_bankroll - self.initial_bankroll) / self.initial_bankroll

    def summary(self) -> str:
        return (
            f"Backtest: {self.movie_name}\n"
            f"  Release: {self.release_date}, Final Score: {self.final_score:.1f}%\n"
            f"  Total Bets: {self.total_bets}, Win Rate: {self.win_rate:.1%}\n"
            f"  PnL: ${self.total_pnl:.2f}, ROI: {self.roi:.1%}\n"
            f"  Final Bankroll: ${self.final_bankroll:.2f}\n"
            f"  Max Drawdown: {self.max_drawdown:.1%}"
        )


def parse_market_threshold(market_name: str) -> int:
    """Extract threshold from market name like 'Above 60' -> 60."""
    parts = market_name.split()
    if len(parts) >= 2 and parts[0].lower() == "above":
        try:
            return int(parts[1])
        except ValueError:
            pass
    raise ValueError(f"Cannot parse threshold from market name: {market_name}")


def load_price_history(csv_path: Path | str) -> pd.DataFrame:
    """
    Load price history CSV into a DataFrame.

    Expected format:
    timestamp,Above 10,Above 15,...,Above 90
    2025-11-13T00:00:00Z,,,38.12,18.54,...

    Prices are in cents (0-100), converted to probability (0-1).
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Convert cents to probability (0-1 scale)
    for col in df.columns:
        df[col] = df[col] / 100.0

    return df


def load_review_history(
    csv_path: Path | str,
    release_date: date,
    date_format: str = "%d-%b",
    year: int | None = None,
) -> pd.DataFrame:
    """
    Load review history CSV and add absolute dates.

    Args:
        csv_path: Path to review CSV
        release_date: The release date for the movie
        date_format: Format of dates in CSV (default "%d-%b" for "7-Oct")
        year: Year to assign to dates (defaults to release_date year)

    Returns:
        DataFrame with columns including parsed_date, sentiment_numeric, etc.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    year = year or release_date.year

    # Parse dates with year
    def parse_date_with_year(date_str: str) -> datetime:
        parsed = datetime.strptime(date_str, date_format)
        return parsed.replace(year=year)

    df["parsed_date"] = df["Date Reviewed"].apply(parse_date_with_year)

    # Map sentiment to numeric
    sentiment_map = {"positive": 1, "negative": 0}
    df["sentiment_numeric"] = df["Review Sentiment"].str.lower().map(sentiment_map)

    # Sort by date and review order
    df = df.sort_values(["parsed_date", "Review Order"]).reset_index(drop=True)

    return df


@dataclass
class ReviewSnapshot:
    """Snapshot of review data at a point in time."""

    current_rating: float  # Percentage positive (0-100)
    num_reviews: int
    days_since_release: int


def compute_rating_at_time(
    review_df: pd.DataFrame,
    cutoff_time: datetime | pd.Timestamp,
) -> ReviewSnapshot:
    """
    Compute the RT rating using only reviews up to cutoff_time.

    Returns:
        ReviewSnapshot with rating details
    """
    # Normalize cutoff_time to match review_df timezone awareness
    if isinstance(cutoff_time, pd.Timestamp):
        # If cutoff is tz-aware and reviews are tz-naive, convert
        if cutoff_time.tz is not None:
            cutoff_time = cutoff_time.tz_localize(None)
    elif isinstance(cutoff_time, datetime):
        # Convert datetime to pandas Timestamp for consistent comparison
        cutoff_time = pd.Timestamp(cutoff_time)
        if cutoff_time.tz is not None:
            cutoff_time = cutoff_time.tz_localize(None)

    # Filter to reviews before or at cutoff
    mask = review_df["parsed_date"] <= cutoff_time
    subset = review_df[mask]

    if len(subset) == 0:
        return ReviewSnapshot(
            current_rating=0.0,
            num_reviews=0,
            days_since_release=0,
        )

    num_reviews = len(subset)
    positive_count = int(subset["sentiment_numeric"].sum())
    current_rating = (positive_count / num_reviews) * 100

    first_review_date = review_df["parsed_date"].min()
    days_since_release = (cutoff_time - first_review_date).days

    return ReviewSnapshot(
        current_rating=current_rating,
        num_reviews=num_reviews,
        days_since_release=max(0, days_since_release),
    )


def kelly_bet_size(prob_win: float, odds: float, kelly_fraction: float = 1.0) -> float:
    """
    Calculate Kelly criterion bet size.

    For binary markets where you pay `price` for a contract worth 1 if you win:
    - If betting YES at price p: odds = (1-p)/p, you win (1-p) on price p
    - If betting NO at price p: odds = p/(1-p), you win p on price (1-p)

    Kelly formula: f* = (p * b - q) / b
    where p = prob of winning, q = 1-p, b = odds (net gain per dollar risked)

    Args:
        prob_win: Model's probability of winning the bet
        odds: Net payout per dollar risked (e.g., if you risk $1 to win $2, odds=2)
        kelly_fraction: Fraction of Kelly to use (default 1.0 = full Kelly)

    Returns:
        Recommended bet size as fraction of bankroll (0 if negative edge)
    """
    if prob_win <= 0 or prob_win >= 1 or odds <= 0:
        return 0.0

    q = 1 - prob_win
    kelly = (prob_win * odds - q) / odds

    # Don't bet if negative edge
    if kelly <= 0:
        return 0.0

    return kelly * kelly_fraction


class Backtester:
    """
    Runs backtests on movie prediction markets using historical data.

    Prevents data leakage by:
    1. Only using review data available up to each timestamp
    2. Not using future price information
    3. Not using the final score until settlement
    """

    def __init__(
        self,
        predictor: BasePredictor,
        config: BacktestConfig | None = None,
    ):
        self.predictor = predictor
        self.config = config or BacktestConfig()

    def _get_model_probability(
        self, threshold: int, features: dict[str, float]
    ) -> float:
        """
        Get model's probability that final score will be above threshold.

        This uses the predictor to get score distribution and calculates
        the probability of exceeding the threshold.
        """
        # Try to get direct score prediction first
        score = self.predictor.predict_score(features)

        if score is not None:
            # Regression model - use score directly
            # Estimate probability based on distance from threshold
            # This is a simple sigmoid-based approach
            # The further above/below threshold, the more confident
            diff = score - threshold
            # Use a sigmoid with reasonable spread (10 points = ~73% confidence)
            prob = 1 / (1 + np.exp(-diff / 10))
            return float(prob)

        # Classification model - use bucket probabilities
        probs = self.predictor.predict_probabilities(features)

        # Map buckets to score ranges:
        # Bucket 0: < 60
        # Bucket 1: 60-90
        # Bucket 2: > 90
        prob_above = 0.0

        if threshold < 60:
            # Above threshold includes buckets 0 (partial), 1, and 2
            # Estimate what fraction of bucket 0 is above threshold
            bucket_0_above = max(0, (60 - threshold) / 60)  # Linear approximation
            prob_above = (
                probs.get(0, 0) * bucket_0_above + probs.get(1, 0) + probs.get(2, 0)
            )
        elif threshold < 90:
            # Above threshold includes bucket 1 (partial) and 2
            bucket_1_above = max(0, (90 - threshold) / 30)  # Bucket 1 spans 60-90
            prob_above = probs.get(1, 0) * bucket_1_above + probs.get(2, 0)
        else:
            # Above threshold is only bucket 2 (partial)
            bucket_2_above = max(0, (100 - threshold) / 10)  # Bucket 2 spans 90-100
            prob_above = probs.get(2, 0) * bucket_2_above

        return float(np.clip(prob_above, 0.001, 0.999))

    def _evaluate_bet_opportunity(
        self,
        threshold: int,
        model_prob: float,
        market_price: float,
    ) -> tuple[str | None, float, float]:
        """
        Evaluate if there's a betting opportunity.

        Returns:
            (direction, edge, kelly_size) or (None, 0, 0) if no bet
        """
        if pd.isna(market_price) or market_price <= 0 or market_price >= 1:
            return None, 0.0, 0.0

        # Calculate edges for YES and NO
        yes_edge = model_prob - market_price
        no_edge = (1 - model_prob) - (1 - market_price)  # = market_price - model_prob

        # Check YES opportunity
        if yes_edge >= self.config.min_edge_threshold:
            # Odds for YES bet: win (1-price) on risk of price
            odds = (1 - market_price) / market_price
            kelly = kelly_bet_size(model_prob, odds, self.config.kelly_fraction)
            return "YES", yes_edge, kelly

        # Check NO opportunity
        if no_edge >= self.config.min_edge_threshold:
            # Odds for NO bet: win price on risk of (1-price)
            odds = market_price / (1 - market_price)
            kelly = kelly_bet_size(1 - model_prob, odds, self.config.kelly_fraction)
            return "NO", no_edge, kelly

        return None, 0.0, 0.0

    def run(
        self,
        movie_name: str,
        review_df: pd.DataFrame,
        price_df: pd.DataFrame,
        release_date: date,
        final_score: float,
    ) -> BacktestResult:
        """
        Run backtest on a single movie.

        Args:
            movie_name: Name of the movie for reporting
            review_df: DataFrame with review history (from load_review_history)
            price_df: DataFrame with price history (from load_price_history)
            release_date: Movie release date
            final_score: Actual final RT score (for settlement)

        Returns:
            BacktestResult with complete performance metrics
        """
        result = BacktestResult(
            movie_name=movie_name,
            release_date=release_date,
            final_score=final_score,
            initial_bankroll=self.config.initial_bankroll,
            final_bankroll=self.config.initial_bankroll,
            peak_bankroll=self.config.initial_bankroll,
        )

        bankroll = self.config.initial_bankroll
        result.bankroll_history.append((price_df.index[0], bankroll))

        # Track pending bets for settlement
        pending_bets: list[BetRecord] = []

        # Iterate through each timestamp in price history
        for timestamp in price_df.index:
            # Compute features using only data available at this time
            # CRITICAL: This prevents data leakage
            snapshot = compute_rating_at_time(review_df, timestamp)

            # Skip if no reviews yet
            if snapshot.num_reviews == 0:
                continue

            # Build standard features
            features = {
                "days_since_release": float(snapshot.days_since_release),
                "current_rating": snapshot.current_rating,
                "num_reviews": float(snapshot.num_reviews),
            }

            # Evaluate each market
            for market_col in price_df.columns:
                try:
                    threshold = parse_market_threshold(market_col)
                except ValueError:
                    continue

                market_price = price_df.loc[timestamp, market_col]

                if pd.isna(market_price):
                    continue

                # Get model's probability for this threshold
                model_prob = self._get_model_probability(threshold, features)

                # Evaluate betting opportunity
                direction, edge, kelly_size = self._evaluate_bet_opportunity(
                    threshold, model_prob, market_price
                )

                if direction is None:
                    continue

                # Apply bet size constraints
                actual_kelly = min(kelly_size, self.config.max_bet_fraction)
                bet_amount = bankroll * actual_kelly

                # Apply minimum bet size
                if bet_amount < self.config.min_bet_size:
                    continue

                # Apply transaction costs
                bet_amount_after_costs = bet_amount * (1 - self.config.transaction_cost)

                # Record the bet
                bet = BetRecord(
                    timestamp=timestamp,
                    market=market_col,
                    direction=direction,
                    model_prob=model_prob,
                    market_price=market_price,
                    edge=edge,
                    kelly_size=kelly_size,
                    actual_bet_size=actual_kelly,
                    bet_amount=bet_amount_after_costs,
                )

                pending_bets.append(bet)

                # Deduct bet amount from bankroll
                bankroll -= bet_amount

                logger.debug(
                    f"Bet placed: {direction} {market_col} @ {market_price:.2f}, "
                    f"model_prob={model_prob:.3f}, edge={edge:.3f}, "
                    f"amount=${bet_amount:.2f}"
                )

            # Update bankroll history
            result.bankroll_history.append((timestamp, bankroll))

        # Settle all bets using final score
        for bet in pending_bets:
            threshold = parse_market_threshold(bet.market)
            actual_above = final_score > threshold

            if bet.direction == "YES":
                bet.outcome = actual_above
                if actual_above:
                    # Win: receive payout (bet_amount / price gives contract count)
                    contracts = bet.bet_amount / bet.market_price
                    payout = contracts * 1.0  # Contract pays $1 if above
                    bet.pnl = payout - bet.bet_amount
                else:
                    bet.pnl = -bet.bet_amount
            else:  # NO
                bet.outcome = not actual_above
                if not actual_above:
                    # Win: receive payout
                    contracts = bet.bet_amount / (1 - bet.market_price)
                    payout = contracts * 1.0
                    bet.pnl = payout - bet.bet_amount
                else:
                    bet.pnl = -bet.bet_amount

            # Add PnL to bankroll (we already deducted the bet amount)
            if bet.outcome:
                # Add back the bet amount plus profit
                bankroll += bet.bet_amount + bet.pnl
            # If lost, bet amount already deducted, pnl is just -bet_amount

            result.bets.append(bet)

            if bet.outcome:
                result.winning_bets += 1
            else:
                result.losing_bets += 1

        # Calculate final metrics
        result.total_bets = len(result.bets)
        result.total_pnl = sum(bet.pnl for bet in result.bets if bet.pnl is not None)
        result.final_bankroll = bankroll

        # Calculate max drawdown
        peak = result.initial_bankroll
        max_dd = 0.0
        for _, br in result.bankroll_history:
            if br > peak:
                peak = br
            dd = (peak - br) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        result.max_drawdown = max_dd
        result.peak_bankroll = peak

        return result


def run_backtest(
    movie_name: str,
    review_csv: Path | str,
    price_csv: Path | str,
    release_date: date,
    final_score: float,
    model_path: Path | str | None = None,
    config: BacktestConfig | None = None,
    date_format: str = "%d-%b",
    review_year: int | None = None,
) -> BacktestResult:
    """
    Convenience function to run a complete backtest.

    Args:
        movie_name: Name of the movie
        review_csv: Path to review history CSV
        price_csv: Path to price history CSV
        release_date: Movie release date
        final_score: Actual final RT score
        model_path: Path to prediction model (uses default if None)
        config: Backtest configuration
        date_format: Format of dates in review CSV
        review_year: Year to assign to review dates

    Returns:
        BacktestResult with complete metrics
    """
    if config is None:
        config = BacktestConfig()

    # Load predictor
    if model_path:
        predictor: BasePredictor = MLPredictor(Path(model_path))
    else:
        # Try to load default model
        default_path = (
            Path(__file__).parent.parent / "models" / "prediction_dummy.model"
        )
        if default_path.exists():
            predictor = MLPredictor(default_path)
        else:
            logger.warning("No model found, using DummyPredictor")
            predictor = DummyPredictor()

    # Load data
    review_df = load_review_history(review_csv, release_date, date_format, review_year)
    price_df = load_price_history(price_csv)

    # Run backtest
    backtester = Backtester(predictor, config)
    return backtester.run(movie_name, review_df, price_df, release_date, final_score)


def run_backtest_from_dataframes(
    movie_name: str,
    review_df: pd.DataFrame,
    price_df: pd.DataFrame,
    release_date: date,
    final_score: float,
    predictor: BasePredictor | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run backtest from pre-loaded DataFrames.

    Useful when you already have the data loaded or want to use
    custom preprocessing.

    Args:
        movie_name: Name of the movie
        review_df: DataFrame with parsed_date and sentiment_numeric columns
        price_df: DataFrame indexed by timestamp with market columns
        release_date: Movie release date
        final_score: Actual final RT score
        predictor: Predictor to use (uses DummyPredictor if None)
        config: Backtest configuration

    Returns:
        BacktestResult with complete metrics
    """
    if predictor is None:
        predictor = DummyPredictor()

    backtester = Backtester(predictor, config)
    return backtester.run(movie_name, review_df, price_df, release_date, final_score)


if __name__ == "__main__":
    # Example usage
    import sys

    logging.basicConfig(level=logging.INFO)

    # Example with command line args
    if len(sys.argv) >= 5:
        movie_name = sys.argv[1]
        review_csv = sys.argv[2]
        price_csv = sys.argv[3]
        release_date = date.fromisoformat(sys.argv[4])
        final_score = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0

        result = run_backtest(
            movie_name=movie_name,
            review_csv=review_csv,
            price_csv=price_csv,
            release_date=release_date,
            final_score=final_score,
        )

        print(result.summary())
        print(f"\nBets placed: {len(result.bets)}")
        for bet in result.bets[:10]:  # Show first 10 bets
            outcome_str = "WIN" if bet.outcome else "LOSS"
            print(
                f"  {bet.timestamp.date()} {bet.direction} {bet.market} "
                f"@ {bet.market_price:.2f} -> {outcome_str} ${bet.pnl:.2f}"
            )
    else:
        print(
            "Usage: python backtest.py <movie_name> <review_csv> <price_csv> <release_date> [final_score]"
        )
        print(
            "Example: python backtest.py 'Moana 2' reviews.csv prices.csv 2025-11-27 75.0"
        )
