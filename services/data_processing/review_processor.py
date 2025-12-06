"""
Configurable review dataset processor.
Decouples transformation logic from specific column names.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnMapping:
    """
    Maps generic field names to actual column names in the CSV.

    This allows the ReviewProcessor to work with datasets that have
    different column naming conventions.
    """

    review_order: str = "Review Order"
    critic_name: str = "Critic Name"
    date_reviewed: str = "Date Reviewed"
    review_sentiment: str = "Review Sentiment"

    # Date format for parsing (e.g., "%d-%b" for "01-Jan")
    date_format: str = "%d-%b"

    # Sentiment value mappings
    positive_value: str = "positive"
    negative_value: str = "negative"


class ReviewProcessor:
    """
    Processes review datasets with configurable column mappings.

    Validates input schemas and transforms raw review data into
    training-ready feature rows.
    """

    def __init__(self, column_mapping: ColumnMapping | None = None):
        """
        Initialize the processor with a column mapping.

        Args:
            column_mapping: ColumnMapping instance. If None, uses default mapping.
        """
        self.mapping = column_mapping or ColumnMapping()

    def validate_and_load(self, csv_path: Path) -> pd.DataFrame:
        """
        Load and validate a review CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Validated and sorted DataFrame

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        return self.validate_and_load_df(df)

    def validate_and_load_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a review DataFrame.

        Args:
            df: DataFrame with review data

        Returns:
            Validated and sorted DataFrame

        Raises:
            ValueError: If required columns are missing or data is invalid
        """

        # Check required columns
        required_cols = {
            self.mapping.review_order,
            self.mapping.critic_name,
            self.mapping.date_reviewed,
            self.mapping.review_sentiment,
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Parse and validate dates
        df["parsed_date"] = pd.to_datetime(
            df[self.mapping.date_reviewed],
            format=self.mapping.date_format,
            errors="coerce"
        )
        if df["parsed_date"].isna().any():
            bad_rows = df[df["parsed_date"].isna()]
            raise ValueError(f"Failed to parse dates in rows: {bad_rows.index.tolist()}")

        # Sort by date and order
        df = df.sort_values(["parsed_date", self.mapping.review_order]).reset_index(drop=True)

        # Map sentiment to numeric
        sentiment_map = {
            self.mapping.positive_value: 1,
            self.mapping.negative_value: 0,
        }
        df["sentiment_numeric"] = df[self.mapping.review_sentiment].str.lower().map(sentiment_map)
        if df["sentiment_numeric"].isna().any():
            bad_rows = df[df["sentiment_numeric"].isna()]
            raise ValueError(f"Unexpected sentiment values in rows: {bad_rows.index.tolist()}")

        return df

    def build_prefix_features(
        self,
        df: pd.DataFrame,
        prefix_lengths: Iterable[int]
    ) -> List[dict]:
        """
        Build feature rows from review prefixes.

        For each prefix length, computes features based on the first N reviews
        and the final rating across all reviews.

        Features are mapped to the standard inference format:
        - days_since_release: days from first review to this prefix point
        - current_rating: the rating based on prefix reviews (pos_ratio)
        - num_reviews: number of reviews in prefix
        - final_score: the final rating (target)

        Args:
            df: Validated DataFrame from validate_and_load
            prefix_lengths: Iterable of prefix sizes to generate

        Returns:
            List of feature dictionaries ready for DataFrame conversion
        """
        total_reviews = len(df)
        total_positive = df["sentiment_numeric"].sum()
        final_rating = (total_positive / total_reviews) * 100

        # Get the release date (first review date)
        release_date = df["parsed_date"].min()

        rows: List[dict] = []
        for n in prefix_lengths:
            if n >= total_reviews:
                continue
            prefix = df.head(n)
            pos = prefix["sentiment_numeric"].sum()
            pos_ratio = (pos / n) * 100

            # Days since release (first review) to last review in prefix
            last_review_date = prefix["parsed_date"].max()
            days_since_release = (last_review_date - release_date).days

            # Map to standard inference feature names
            rows.append(
                {
                    "days_since_release": float(days_since_release),
                    "current_rating": pos_ratio,
                    "num_reviews": float(n),
                    "final_score": final_rating,
                }
            )
        return rows

    def process(
        self,
        input_path: Path,
        output_path: Path,
        prefix_lengths: Iterable[int] = (3, 5, 10, 20, 30, 40, 50),
    ) -> Path:
        """
        Full processing pipeline: load, validate, transform, and save.

        Args:
            input_path: Path to input CSV
            output_path: Path to output processed CSV
            prefix_lengths: Prefix sizes to generate

        Returns:
            Path to the output file

        Raises:
            ValueError: If validation or processing fails
        """
        logger.info("Loading review data from %s", input_path)
        df = self.validate_and_load(input_path)

        logger.info("Building feature rows for prefix lengths: %s", prefix_lengths)
        rows = self.build_prefix_features(df, prefix_lengths)

        if not rows:
            raise ValueError("No rows produced; check prefix lengths vs. dataset size.")

        out_df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        logger.info("Wrote %d rows to %s", len(out_df), output_path.resolve())

        return output_path
