#!/usr/bin/env python3
"""
Generate synthetic Rotten Tomatoes movie rating data.

Simulates realistic RT score evolution patterns:
- Scores converge to final value over ~4 days (configurable)
- Early scores tend to be lower than final (critics who like it review later)
- ~50% of reviews come on day 1, rest stream in exponentially until day 10
- Introduces realistic noise

Usage:
    python scripts/generate_synthetic_data.py --help
    python scripts/generate_synthetic_data.py --num-movies 500 --output data/processed_dummy.csv
    python scripts/generate_synthetic_data.py --config scripts/datagen_config.json
"""

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DataGenConfig:
    """Configuration for synthetic data generation."""

    # Dataset size
    num_movies: int = 500
    observations_per_movie: int = 15  # Number of snapshots per movie

    # Score distribution
    min_final_score: float = 5.0
    max_final_score: float = 98.0
    score_distribution: str = "uniform"  # "uniform", "bimodal", or "normal"
    bimodal_peaks: Tuple[float, float] = (25.0, 75.0)
    bimodal_std: float = 15.0
    normal_mean: float = 60.0
    normal_std: float = 20.0

    # Score convergence dynamics
    convergence_days: float = 4.0  # Days for score to ~95% converge to final
    early_score_bias: float = -8.0  # Early scores tend to be this much lower
    early_bias_decay_rate: float = 0.5  # How fast the bias decays (per day)

    # Review accumulation dynamics
    day1_review_fraction: float = 0.5  # Fraction of reviews on day 1
    review_accumulation_rate: float = 0.3  # Exponential rate for remaining reviews
    max_review_days: int = 10  # Days until all reviews are in
    min_total_reviews: int = 20
    max_total_reviews: int = 400

    # Observation window
    min_days_since_release: int = 0
    max_days_since_release: int = 14

    # Noise parameters
    score_noise_std: float = 5.0  # Noise in observed score
    review_count_noise_std: float = 0.1  # Proportional noise in review count

    # Bucket thresholds (for classification target)
    bucket_thresholds: List[float] = field(default_factory=lambda: [60.0, 90.0])

    # Random seed
    seed: int = 42

    @classmethod
    def from_json(cls, path: str) -> "DataGenConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Convert tuple fields
        if "bimodal_peaks" in data:
            data["bimodal_peaks"] = tuple(data["bimodal_peaks"])
        if "bucket_thresholds" in data:
            data["bucket_thresholds"] = list(data["bucket_thresholds"])
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        data = {
            "num_movies": self.num_movies,
            "observations_per_movie": self.observations_per_movie,
            "min_final_score": self.min_final_score,
            "max_final_score": self.max_final_score,
            "score_distribution": self.score_distribution,
            "bimodal_peaks": list(self.bimodal_peaks),
            "bimodal_std": self.bimodal_std,
            "normal_mean": self.normal_mean,
            "normal_std": self.normal_std,
            "convergence_days": self.convergence_days,
            "early_score_bias": self.early_score_bias,
            "early_bias_decay_rate": self.early_bias_decay_rate,
            "day1_review_fraction": self.day1_review_fraction,
            "review_accumulation_rate": self.review_accumulation_rate,
            "max_review_days": self.max_review_days,
            "min_total_reviews": self.min_total_reviews,
            "max_total_reviews": self.max_total_reviews,
            "min_days_since_release": self.min_days_since_release,
            "max_days_since_release": self.max_days_since_release,
            "score_noise_std": self.score_noise_std,
            "review_count_noise_std": self.review_count_noise_std,
            "bucket_thresholds": self.bucket_thresholds,
            "seed": self.seed,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved config to %s", path)


def generate_final_score(config: DataGenConfig, rng: random.Random) -> float:
    """Generate a final RT score based on the configured distribution."""
    if config.score_distribution == "uniform":
        return rng.uniform(config.min_final_score, config.max_final_score)

    elif config.score_distribution == "bimodal":
        # Pick one of two peaks
        peak = rng.choice(config.bimodal_peaks)
        score = rng.gauss(peak, config.bimodal_std)

    elif config.score_distribution == "normal":
        score = rng.gauss(config.normal_mean, config.normal_std)

    else:
        raise ValueError(f"Unknown distribution: {config.score_distribution}")

    # Clamp to valid range
    return max(config.min_final_score, min(config.max_final_score, score))


def compute_review_count(
    days: float, total_reviews: int, config: DataGenConfig, rng: random.Random
) -> int:
    """
    Compute how many reviews have accumulated by a given day.

    Model:
    - Day 1: ~50% of reviews
    - Days 2-10: Remaining reviews accumulate exponentially
    - After day 10: All reviews in
    """
    if days <= 0:
        return 0

    if days >= config.max_review_days:
        base_count = total_reviews
    elif days <= 1:
        # Linear ramp on day 1
        base_count = int(total_reviews * config.day1_review_fraction * days)
    else:
        # Day 1 reviews
        day1_reviews = int(total_reviews * config.day1_review_fraction)
        remaining_reviews = total_reviews - day1_reviews

        # Exponential accumulation of remaining reviews
        # At day=1, we have day1_reviews
        # At day=max_review_days, we have all reviews
        days_since_day1 = days - 1
        max_days_for_remaining = config.max_review_days - 1

        # Exponential: reviews(t) = remaining * (1 - e^(-rate * t)) / (1 - e^(-rate * max_t))
        rate = config.review_accumulation_rate
        if rate > 0:
            progress = (1 - math.exp(-rate * days_since_day1)) / (
                1 - math.exp(-rate * max_days_for_remaining)
            )
        else:
            progress = days_since_day1 / max_days_for_remaining

        accumulated_remaining = int(remaining_reviews * progress)
        base_count = day1_reviews + accumulated_remaining

    # Add noise
    noise = rng.gauss(0, config.review_count_noise_std * base_count)
    noisy_count = int(base_count + noise)

    return max(1, min(total_reviews, noisy_count))


def compute_observed_score(
    days: float, final_score: float, config: DataGenConfig, rng: random.Random
) -> float:
    """
    Compute the observed RT score at a given day.

    Model:
    - Early scores are biased lower (early reviewers tend to be harsher)
    - Scores converge to final over ~4 days
    - Add observation noise
    """
    # Early bias that decays over time
    # bias(t) = early_bias * e^(-decay_rate * t)
    bias = config.early_score_bias * math.exp(-config.early_bias_decay_rate * days)

    # Convergence: score approaches final exponentially
    # convergence_factor goes from 0 to 1 as days increases
    # At convergence_days, we want ~95% convergence, so rate = 3/convergence_days
    rate = 3.0 / config.convergence_days
    convergence_factor = 1 - math.exp(-rate * days)

    # The "true" score at this time (before noise)
    # Starts at final_score + early_bias, converges to final_score
    true_score = final_score + bias * (1 - convergence_factor)

    # Add observation noise
    noise = rng.gauss(0, config.score_noise_std)
    observed = true_score + noise

    # Clamp to valid range
    return max(0.0, min(100.0, observed))


def score_to_bucket(score: float, thresholds: List[float]) -> int:
    """Convert a score to a bucket index based on thresholds."""
    for i, threshold in enumerate(thresholds):
        if score < threshold:
            return i
    return len(thresholds)


def generate_dataset(config: DataGenConfig) -> pd.DataFrame:
    """Generate the full synthetic dataset."""
    rng = random.Random(config.seed)

    records = []

    for movie_idx in range(config.num_movies):
        # Generate movie properties
        final_score = generate_final_score(config, rng)
        total_reviews = rng.randint(config.min_total_reviews, config.max_total_reviews)
        final_bucket = score_to_bucket(final_score, config.bucket_thresholds)

        # Generate observations at different days
        for obs_idx in range(config.observations_per_movie):
            # Sample a random day for this observation
            days = rng.uniform(
                config.min_days_since_release, config.max_days_since_release
            )

            # Compute observed values at this day
            num_reviews = compute_review_count(days, total_reviews, config, rng)
            current_rating = compute_observed_score(days, final_score, config, rng)

            records.append(
                {
                    "movie_id": movie_idx,
                    "days_since_release": round(days, 2),
                    "current_rating": round(current_rating, 2),
                    "num_reviews": num_reviews,
                    "final_score": round(final_score, 2),
                    "final_rating_bucket": final_bucket,
                }
            )

    df = pd.DataFrame(records)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Rotten Tomatoes data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with defaults
  python scripts/generate_synthetic_data.py

  # Generate more movies with bimodal distribution
  python scripts/generate_synthetic_data.py --num-movies 1000 --score-distribution bimodal

  # Use a config file
  python scripts/generate_synthetic_data.py --config my_config.json

  # Save current config as template
  python scripts/generate_synthetic_data.py --save-config template_config.json
        """,
    )

    # Config file options
    parser.add_argument(
        "--config", type=str, help="Load configuration from JSON file"
    )
    parser.add_argument(
        "--save-config", type=str, help="Save configuration to JSON file and exit"
    )

    # Dataset size
    parser.add_argument(
        "--num-movies", type=int, default=500, help="Number of movies to generate"
    )
    parser.add_argument(
        "--observations-per-movie",
        type=int,
        default=15,
        help="Number of observation snapshots per movie",
    )

    # Score distribution
    parser.add_argument(
        "--score-distribution",
        choices=["uniform", "bimodal", "normal"],
        default="uniform",
        help="Distribution of final scores",
    )
    parser.add_argument(
        "--min-final-score", type=float, default=5.0, help="Minimum final score"
    )
    parser.add_argument(
        "--max-final-score", type=float, default=98.0, help="Maximum final score"
    )

    # Dynamics
    parser.add_argument(
        "--convergence-days",
        type=float,
        default=4.0,
        help="Days for score to ~95%% converge to final",
    )
    parser.add_argument(
        "--early-bias",
        type=float,
        default=-8.0,
        help="Early score bias (negative = starts lower)",
    )
    parser.add_argument(
        "--day1-review-fraction",
        type=float,
        default=0.5,
        help="Fraction of reviews on day 1",
    )

    # Review counts
    parser.add_argument(
        "--min-reviews", type=int, default=20, help="Minimum total reviews per movie"
    )
    parser.add_argument(
        "--max-reviews", type=int, default=400, help="Maximum total reviews per movie"
    )

    # Noise
    parser.add_argument(
        "--score-noise", type=float, default=5.0, help="Std dev of score observation noise"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_dummy.csv",
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load or build config
    if args.config:
        config = DataGenConfig.from_json(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        config = DataGenConfig(
            num_movies=args.num_movies,
            observations_per_movie=args.observations_per_movie,
            score_distribution=args.score_distribution,
            min_final_score=args.min_final_score,
            max_final_score=args.max_final_score,
            convergence_days=args.convergence_days,
            early_score_bias=args.early_bias,
            day1_review_fraction=args.day1_review_fraction,
            min_total_reviews=args.min_reviews,
            max_total_reviews=args.max_reviews,
            score_noise_std=args.score_noise,
            seed=args.seed,
        )

    # Save config if requested
    if args.save_config:
        config.to_json(args.save_config)
        return

    # Generate data
    logger.info("Generating synthetic data with config:")
    logger.info("  - %d movies x %d observations = %d rows",
                config.num_movies, config.observations_per_movie,
                config.num_movies * config.observations_per_movie)
    logger.info("  - Score distribution: %s", config.score_distribution)
    logger.info("  - Convergence: ~%.1f days", config.convergence_days)
    logger.info("  - Early bias: %.1f", config.early_score_bias)
    logger.info("  - Day 1 reviews: %.0f%%", config.day1_review_fraction * 100)

    df = generate_dataset(config)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save version without movie_id and final_score for training
    train_df = df[["days_since_release", "current_rating", "num_reviews", "final_rating_bucket"]]
    train_df.to_csv(output_path, index=False)
    logger.info("Saved training data to %s (%d rows)", output_path, len(train_df))

    # Also save full data for analysis
    full_output = output_path.with_suffix(".full.csv")
    df.to_csv(full_output, index=False)
    logger.info("Saved full data (with movie_id, final_score) to %s", full_output)

    # Print summary stats
    logger.info("\nDataset Summary:")
    logger.info("  Final score range: %.1f - %.1f (mean: %.1f)",
                df["final_score"].min(), df["final_score"].max(), df["final_score"].mean())
    logger.info("  Bucket distribution:")
    bucket_counts = df["final_rating_bucket"].value_counts().sort_index()
    for bucket, count in bucket_counts.items():
        pct = count / len(df) * 100
        logger.info("    Bucket %d: %d (%.1f%%)", bucket, count, pct)


if __name__ == "__main__":
    main()
