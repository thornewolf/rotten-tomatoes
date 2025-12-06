"""
Example scraper implementation demonstrating the BaseScraper interface.

This is a reference implementation showing how to:
1. Inherit from BaseScraper
2. Implement required methods
3. Handle data transformation
4. Use helper utilities

DO NOT use this for actual scraping - it generates synthetic data.
"""

import random
import re
from datetime import datetime, timedelta
from typing import Iterator

from scripts.scrapers.base import (
    BaseScraper,
    Review,
    ScraperConfig,
    normalize_sentiment,
    parse_date_to_standard,
)


class ExampleScraper(BaseScraper):
    """
    Example scraper that generates synthetic review data.

    This demonstrates the expected scraper interface without
    making actual HTTP requests. Use as a template for real scrapers.
    """

    def get_movie_id(self, source: str) -> str:
        """
        Extract movie ID from a source string.

        For real scrapers, this would parse a URL like:
            https://www.rottentomatoes.com/m/example_movie
        to extract "example_movie"
        """
        # Handle URL-like input
        if "/" in source:
            # Extract last path segment
            return source.rstrip("/").split("/")[-1]
        # Assume it's already an ID
        return source

    def get_output_filename(self, movie_id: str) -> str:
        """Generate output filename."""
        # Sanitize the movie ID for filesystem use
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", movie_id)
        return f"{safe_id}-reviews.csv"

    def fetch_reviews(self, source: str) -> Iterator[Review]:
        """
        Generate synthetic reviews for demonstration.

        In a real scraper, this would:
        1. Make HTTP requests to the source
        2. Parse HTML/JSON responses
        3. Transform to Review objects
        4. Handle pagination
        5. Respect rate limits
        """
        # Seed with movie ID for reproducibility
        movie_id = self.get_movie_id(source)
        random.seed(hash(movie_id) % (2**32))

        # Generate a random number of reviews
        num_reviews = random.randint(50, 200)

        # Generate release date (within last year)
        base_date = datetime.now() - timedelta(days=random.randint(30, 365))

        # Critics pool
        critics = [
            "Roger Ebert", "A.O. Scott", "Peter Travers",
            "David Ehrlich", "Richard Roeper", "Manohla Dargis",
            "Owen Gleiberman", "Peter Bradshaw", "Mark Kermode",
            "Claudia Puig", "Lisa Schwarzbaum", "Todd McCarthy",
        ]

        # Generate reviews over time
        current_date = base_date
        for i in range(1, num_reviews + 1):
            # Advance date (more reviews early on)
            if i <= num_reviews * 0.5:
                # First 50% of reviews in first 2 days
                hours_advance = random.randint(1, 12)
            else:
                # Rest spread over next 5 days
                hours_advance = random.randint(2, 24)

            current_date += timedelta(hours=hours_advance)

            # Random sentiment (could be biased based on movie_id)
            sentiment = "positive" if random.random() > 0.4 else "negative"

            yield Review(
                review_order=i,
                critic_name=random.choice(critics),
                date_reviewed=current_date.strftime("%d-%b"),
                review_sentiment=sentiment,
            )


class CSVImportScraper(BaseScraper):
    """
    Scraper that imports from an existing CSV with different column names.

    Use this as a template for converting data from other formats.
    """

    def __init__(
        self,
        config: ScraperConfig | None = None,
        column_mapping: dict | None = None,
        date_format: str = "%Y-%m-%d",
    ):
        """
        Initialize with column mapping.

        Args:
            config: Scraper configuration
            column_mapping: Map from source columns to standard columns
                e.g., {"rating": "Review Sentiment", "reviewer": "Critic Name"}
            date_format: strptime format for parsing dates
        """
        super().__init__(config)
        self.column_mapping = column_mapping or {}
        self.date_format = date_format

    def get_movie_id(self, source: str) -> str:
        """Extract movie ID from CSV filename."""
        from pathlib import Path
        return Path(source).stem

    def get_output_filename(self, movie_id: str) -> str:
        return f"{movie_id}-reviews.csv"

    def fetch_reviews(self, source: str) -> Iterator[Review]:
        """Read and transform reviews from source CSV."""
        import csv
        from pathlib import Path

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source CSV not found: {source}")

        with open(source_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader, start=1):
                # Apply column mapping
                mapped = {}
                for src_col, val in row.items():
                    target_col = self.column_mapping.get(src_col, src_col)
                    mapped[target_col] = val

                # Extract and transform fields
                date_str = mapped.get("Date Reviewed", "")
                if date_str and self.date_format != "%d-%b":
                    date_str = parse_date_to_standard(date_str, self.date_format)

                sentiment = mapped.get("Review Sentiment", "")
                if sentiment:
                    sentiment = normalize_sentiment(sentiment)

                yield Review(
                    review_order=int(mapped.get("Review Order", i)),
                    critic_name=mapped.get("Critic Name", f"Critic {i}"),
                    date_reviewed=date_str,
                    review_sentiment=sentiment,
                )


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create scraper with custom config
    config = ScraperConfig(
        output_dir=Path("data"),
        max_reviews=100,
    )

    scraper = ExampleScraper(config)

    # Scrape a "movie"
    output = scraper.scrape("example_movie")
    print(f"Generated: {output}")
