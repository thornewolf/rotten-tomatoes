import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

from services.data_processing import ColumnMapping, ReviewProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transform_tron(
    tron_path: str = "data/tron-sample.csv",
    out_path: str = "data/processed.csv",
    prefix_lengths: Iterable[int] = (3, 5, 10, 20, 30, 40, 50),
    column_mapping: Optional[Dict[str, str]] = None,
):
    """
    Build a dataset to predict the final average rating (all critics) from the early critics.

    Each row uses only the first N critics' sentiments (for various N) to predict the final
    average rating across all critics.

    Args:
        tron_path: Path to input CSV
        out_path: Path to output processed CSV
        prefix_lengths: Iterable of prefix sizes to generate
        column_mapping: Optional dictionary to override default column mappings.
                       Keys: review_order, critic_name, date_reviewed, review_sentiment,
                            date_format, positive_value, negative_value

    Returns:
        Path to output file
    """
    tron_path = Path(tron_path)
    out_path = Path(out_path)

    # Create ColumnMapping from dict if provided
    if column_mapping:
        mapping = ColumnMapping(**column_mapping)
    else:
        mapping = ColumnMapping()  # Use defaults

    # Use ReviewProcessor to handle the transformation
    processor = ReviewProcessor(column_mapping=mapping)
    return processor.process(tron_path, out_path, prefix_lengths)


if __name__ == "__main__":
    transform_tron()
