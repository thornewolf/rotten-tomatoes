"""
Base scraper interface for collecting RT review data.

All scrapers should inherit from BaseScraper and implement the required methods.
This ensures consistent output format across different data sources.
"""

import csv
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Review:
    """
    A single review in the standardized format.

    This is the canonical representation of a review that all scrapers
    must produce. The fields map directly to the expected CSV columns.
    """

    review_order: int
    critic_name: str
    date_reviewed: str  # Format: "DD-Mon" (e.g., "15-Jan")
    review_sentiment: str  # "positive" or "negative"

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV writing."""
        return {
            "Review Order": self.review_order,
            "Critic Name": self.critic_name,
            "Date Reviewed": self.date_reviewed,
            "Review Sentiment": self.review_sentiment,
        }


@dataclass
class ScraperConfig:
    """
    Configuration for a scraper run.

    Attributes:
        output_dir: Directory to save scraped data
        rate_limit: Seconds between requests (be respectful!)
        max_reviews: Maximum reviews to collect (None = unlimited)
        headers: Custom HTTP headers for requests
    """

    output_dir: Path = field(default_factory=lambda: Path("data"))
    rate_limit: float = 1.0  # seconds between requests
    max_reviews: Optional[int] = None
    headers: dict = field(default_factory=lambda: {
        "User-Agent": "Mozilla/5.0 (compatible; RT-Scraper/1.0)"
    })


class BaseScraper(ABC):
    """
    Abstract base class for RT review scrapers.

    Subclasses must implement:
    - get_movie_id(): Extract movie identifier from URL/input
    - fetch_reviews(): Yield Review objects from the source
    - get_output_filename(): Generate output filename

    The base class provides:
    - Standard CSV output format
    - Rate limiting support
    - Progress logging
    - Error handling
    """

    # Expected CSV columns in output order
    CSV_COLUMNS = ["Review Order", "Critic Name", "Date Reviewed", "Review Sentiment"]

    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize the scraper.

        Args:
            config: Scraper configuration. Uses defaults if not provided.
        """
        self.config = config or ScraperConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_movie_id(self, source: str) -> str:
        """
        Extract a unique movie identifier from the source.

        Args:
            source: URL or identifier for the movie

        Returns:
            Unique string identifier (used in filename)
        """
        pass

    @abstractmethod
    def fetch_reviews(self, source: str) -> Iterator[Review]:
        """
        Fetch reviews from the source.

        Args:
            source: URL or identifier for the movie

        Yields:
            Review objects in chronological order (earliest first)

        Note:
            - Reviews should be yielded in order of review_order
            - review_order should start at 1
            - date_reviewed should be in "DD-Mon" format
            - review_sentiment should be "positive" or "negative"
        """
        pass

    @abstractmethod
    def get_output_filename(self, movie_id: str) -> str:
        """
        Generate the output filename for a movie.

        Args:
            movie_id: Unique movie identifier

        Returns:
            Filename (without directory path)
        """
        pass

    def scrape(self, source: str) -> Path:
        """
        Run the full scrape pipeline.

        Args:
            source: URL or identifier for the movie

        Returns:
            Path to the output CSV file
        """
        movie_id = self.get_movie_id(source)
        filename = self.get_output_filename(movie_id)
        output_path = self.config.output_dir / filename

        logger.info("Scraping %s -> %s", source, output_path)

        reviews: List[Review] = []
        for review in self.fetch_reviews(source):
            reviews.append(review)
            if self.config.max_reviews and len(reviews) >= self.config.max_reviews:
                logger.info("Reached max_reviews limit (%d)", self.config.max_reviews)
                break

        self._write_csv(output_path, reviews)
        logger.info("Wrote %d reviews to %s", len(reviews), output_path)

        return output_path

    def _write_csv(self, path: Path, reviews: List[Review]) -> None:
        """Write reviews to CSV in the standard format."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            for review in reviews:
                writer.writerow(review.to_dict())


def parse_date_to_standard(date_str: str, input_format: str = "%Y-%m-%d") -> str:
    """
    Convert a date string to the standard "DD-Mon" format.

    Args:
        date_str: Input date string
        input_format: strptime format for parsing input

    Returns:
        Date in "DD-Mon" format (e.g., "15-Jan")

    Examples:
        >>> parse_date_to_standard("2024-01-15")
        "15-Jan"
        >>> parse_date_to_standard("January 15, 2024", "%B %d, %Y")
        "15-Jan"
    """
    dt = datetime.strptime(date_str, input_format)
    return dt.strftime("%d-%b")


def normalize_sentiment(value: str) -> str:
    """
    Normalize sentiment values to "positive" or "negative".

    Args:
        value: Raw sentiment value from source

    Returns:
        "positive" or "negative"

    Raises:
        ValueError: If sentiment cannot be determined
    """
    normalized = value.strip().lower()

    positive_values = {"positive", "fresh", "1", "true", "yes", "+"}
    negative_values = {"negative", "rotten", "0", "false", "no", "-"}

    if normalized in positive_values:
        return "positive"
    if normalized in negative_values:
        return "negative"

    raise ValueError(f"Cannot normalize sentiment: {value}")
