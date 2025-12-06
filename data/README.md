# Data Pipeline Guide

This document describes how to collect, process, and use datasets for the RT prediction model.

## Directory Structure

```
data/
├── README.md                    # This file
├── tron-sample.csv              # Raw review data (review-level)
├── processed_tron.csv           # Processed training data
├── processed_dummy.csv          # Synthetic training data
└── <movie>-reviews.csv          # Additional scraped data
```

## Quick Start

```bash
# List available datasets
just list-datasets

# Train with tron dataset (uses existing processed data)
just train

# Train with dummy data (generates synthetic data)
just train-dummy

# Train any dataset by name
just train-dataset tron
```

## Data Formats

### 1. Raw Review Format (`format: reviews`)

Review-level data from scrapers. Each row is one critic review.

**Required columns:**
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Review Order` | int | Chronological position (1-indexed) | `1`, `2`, `3` |
| `Critic Name` | str | Name of the critic | `Roger Ebert` |
| `Date Reviewed` | str | Review date in `DD-Mon` format | `15-Jan`, `03-Feb` |
| `Review Sentiment` | str | `positive` or `negative` | `positive` |

**Example CSV:**
```csv
Review Order,Critic Name,Date Reviewed,Review Sentiment
1,Roger Ebert,15-Jan,positive
2,Peter Travers,15-Jan,negative
3,A.O. Scott,16-Jan,positive
```

### 2. Processed Format (`format: processed`)

Training-ready data. Each row is one observation point.

**Required columns:**
| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `days_since_release` | float | Days from first review | `0.0` - `30.0` |
| `current_rating` | float | Rating at observation point | `0.0` - `100.0` |
| `num_reviews` | float | Number of reviews so far | `1.0` - `500.0` |
| `final_score` | float | Final RT score (target) | `0.0` - `100.0` |

**Example CSV:**
```csv
days_since_release,current_rating,num_reviews,final_score
0.0,33.33,3.0,52.12
0.0,20.0,5.0,52.12
0.0,30.0,10.0,52.12
```

## Dataset Configuration

Datasets are defined in `datasets.yaml` at the project root.

### Adding a New Dataset

1. **Add raw data** to `data/` directory
2. **Define dataset** in `datasets.yaml`:

```yaml
datasets:
  my_movie:
    description: "My new movie data"
    sources:
      - path: data/my_movie-reviews.csv
        format: reviews
        transform: review_prefix
        transform_args:
          prefix_lengths: [3, 5, 10, 20, 30, 40, 50]
    output_path: data/processed_my_movie.csv
    train_test_split: 0.2
    random_state: 42
```

3. **Train the model:**
```bash
just train-dataset my_movie
```

### Configuration Reference

```yaml
datasets:
  <name>:
    description: "Human-readable description"
    sources:
      - path: data/<file>.csv           # Relative to project root
        format: reviews | processed     # Data format
        transform: review_prefix        # Optional transformation
        transform_args:
          prefix_lengths: [3, 5, 10]    # Transform parameters
    output_path: data/processed_<name>.csv
    train_test_split: 0.2               # Test set fraction
    random_state: 42                    # For reproducibility
```

## Writing Scrapers

Scrapers collect review data from external sources. All scrapers inherit from `BaseScraper`.

### Scraper Interface

```python
from scripts.scrapers.base import BaseScraper, Review, ScraperConfig

class MyScraper(BaseScraper):
    def get_movie_id(self, source: str) -> str:
        """Extract movie identifier from URL/input."""
        # Example: "https://example.com/movie/tron" -> "tron"
        return source.split("/")[-1]

    def get_output_filename(self, movie_id: str) -> str:
        """Generate output filename."""
        return f"{movie_id}-reviews.csv"

    def fetch_reviews(self, source: str) -> Iterator[Review]:
        """Yield Review objects from the source."""
        for i, review_data in enumerate(self._fetch_from_api(source)):
            yield Review(
                review_order=i + 1,
                critic_name=review_data["critic"],
                date_reviewed=review_data["date"],  # Must be DD-Mon format
                review_sentiment=review_data["sentiment"],
            )
```

### Helper Utilities

```python
from scripts.scrapers.base import parse_date_to_standard, normalize_sentiment

# Convert dates to standard format
date = parse_date_to_standard("2024-01-15")  # "15-Jan"
date = parse_date_to_standard("January 15, 2024", "%B %d, %Y")  # "15-Jan"

# Normalize sentiment values
sentiment = normalize_sentiment("Fresh")     # "positive"
sentiment = normalize_sentiment("Rotten")    # "negative"
sentiment = normalize_sentiment("1")         # "positive"
```

### Running a Scraper

```python
from pathlib import Path
from scripts.scrapers.base import ScraperConfig
from scripts.scrapers.my_scraper import MyScraper

config = ScraperConfig(
    output_dir=Path("data"),
    rate_limit=1.0,  # seconds between requests
    max_reviews=None,  # no limit
)

scraper = MyScraper(config)
output_path = scraper.scrape("https://example.com/movie/tron")
print(f"Saved to: {output_path}")
```

### Example: Converting Existing Data

If you have data in a different format, use `CSVImportScraper`:

```python
from scripts.scrapers.example_scraper import CSVImportScraper

scraper = CSVImportScraper(
    column_mapping={
        "rating": "Review Sentiment",
        "reviewer": "Critic Name",
        "date": "Date Reviewed",
        "order": "Review Order",
    },
    date_format="%Y-%m-%d",
)

output = scraper.scrape("path/to/source.csv")
```

## Data Transformations

### Review Prefix Transform

Converts raw review data to training features by computing prefix statistics.

**What it does:**
1. Sorts reviews chronologically
2. For each prefix length (3, 5, 10, ...):
   - Computes `current_rating` from first N reviews
   - Computes `days_since_release` from dates
   - Sets `num_reviews` = N
3. Uses final rating across all reviews as `final_score`

**Configuration:**
```yaml
transform: review_prefix
transform_args:
  prefix_lengths: [3, 5, 10, 20, 30, 40, 50]
```

## End-to-End Example

### Step 1: Create Scraper

```python
# scripts/scrapers/rt_scraper.py
from scripts.scrapers.base import BaseScraper, Review

class RTScraper(BaseScraper):
    def get_movie_id(self, source: str) -> str:
        # https://www.rottentomatoes.com/m/tron -> tron
        return source.rstrip("/").split("/")[-1]

    def get_output_filename(self, movie_id: str) -> str:
        return f"{movie_id}-reviews.csv"

    def fetch_reviews(self, source: str) -> Iterator[Review]:
        # Your scraping logic here
        ...
```

### Step 2: Collect Data

```python
from scripts.scrapers.rt_scraper import RTScraper

scraper = RTScraper()
scraper.scrape("https://www.rottentomatoes.com/m/new_movie")
# Saves to: data/new_movie-reviews.csv
```

### Step 3: Configure Dataset

Add to `datasets.yaml`:
```yaml
datasets:
  new_movie:
    description: "New Movie RT reviews"
    sources:
      - path: data/new_movie-reviews.csv
        format: reviews
        transform: review_prefix
        transform_args:
          prefix_lengths: [3, 5, 10, 20, 30, 40, 50]
    output_path: data/processed_new_movie.csv
```

### Step 4: Train Model

```bash
just train-dataset new_movie
```

### Step 5: Use Model

The trained model is saved to `prediction_new_movie.model` and can be loaded:

```python
from services.predictors.ml import MLPredictor

predictor = MLPredictor("prediction_new_movie.model")
score = predictor.predict_score({
    "days_since_release": 2.0,
    "current_rating": 75.0,
    "num_reviews": 30.0,
})
print(f"Predicted final score: {score}")
```

## Combining Datasets

For robust training, combine multiple movie datasets:

```yaml
datasets:
  combined:
    description: "Combined multi-movie dataset"
    sources:
      - path: data/processed_tron.csv
        format: processed
      - path: data/processed_new_movie.csv
        format: processed
      - path: data/processed_another_movie.csv
        format: processed
    output_path: data/processed_combined.csv
```

Train with: `just train-dataset combined`

## Generating Synthetic Data

For development/testing without real data:

```bash
# Generate and train with 600 rows
just train-dummy

# Or specify custom row count
PYTHONPATH=. uv run python scripts/train.py --dataset dummy --generate --dummy-rows 2000
```

The synthetic data generator (`scripts/generate_synthetic_data.py`) creates realistic patterns:
- Score convergence over ~4 days
- Early reviews biased lower
- 50% of reviews on day 1

## Model Registry

Models are configured in `models.yaml`:

```yaml
models:
  tron:
    path: prediction_tron.model
    description: Trained on Tron movie data
    dataset: tron

  dummy:
    path: prediction_dummy.model
    description: Trained on synthetic data
    dataset: dummy
```

The API automatically loads models from this registry.

## Troubleshooting

### "Missing required columns"

Your CSV is missing expected columns. Check:
- Column names match exactly (case-sensitive)
- No BOM issues (use `encoding="utf-8-sig"` when reading)

### "Failed to parse dates"

Date format doesn't match. Ensure:
- Dates are in `DD-Mon` format: `15-Jan`, `03-Feb`
- Month abbreviations are English: Jan, Feb, Mar...

### "X has N features, but model expects M"

Feature mismatch between training and inference. Ensure:
- Training uses standard 3 features: `days_since_release`, `current_rating`, `num_reviews`
- All models use the same feature set

### "Unknown transformer"

The transform name in `datasets.yaml` isn't registered. Built-in transforms:
- `review_prefix`: Convert reviews to prefix features

## API Reference

### Core Classes

- `DatasetConfig`: Configuration for a dataset
- `DatasetRegistry`: Registry of all datasets
- `DatasetLoader`: Loads and processes datasets
- `BaseScraper`: Base class for scrapers
- `Review`: Single review data object

### Files

| File | Description |
|------|-------------|
| `datasets.yaml` | Dataset configurations |
| `core/datasets.py` | Dataset management system |
| `scripts/scrapers/base.py` | Scraper base class |
| `scripts/train.py` | Training script |
| `services/data_processing/review_processor.py` | Review transformation |
