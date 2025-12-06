"""
Unified training script using dataset configurations.

Usage:
    # Train using a configured dataset
    python scripts/train.py --dataset tron
    python scripts/train.py --dataset dummy

    # Generate dummy data first (if needed)
    python scripts/train.py --dataset dummy --generate

    # Override paths
    python scripts/train.py --dataset tron --model-path custom.model
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from core.datasets import (
    DatasetConfig,
    DatasetLoader,
    DatasetRegistry,
    get_dataset_registry,
    STANDARD_FEATURES,
    STANDARD_TARGET,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dummy_data(output_path: Path, rows: int = 600, seed: int = 42) -> Path:
    """
    Generate synthetic RT data using realistic convergence model.

    This creates training data that simulates:
    - Score convergence over ~4 days
    - Early scores biased lower
    - ~50% of reviews on day 1, exponential accumulation after
    """
    from scripts.generate_synthetic_data import DataGenConfig, generate_dataset

    observations_per_movie = 15
    num_movies = max(1, rows // observations_per_movie)

    config = DataGenConfig(
        num_movies=num_movies,
        observations_per_movie=observations_per_movie,
        seed=seed,
        score_distribution="uniform",
        min_final_score=5.0,
        max_final_score=98.0,
        convergence_days=4.0,
        early_score_bias=-8.0,
        day1_review_fraction=0.5,
        min_total_reviews=20,
        max_total_reviews=400,
    )

    df = generate_dataset(config)

    # Keep only training columns
    train_df = df[STANDARD_FEATURES + [STANDARD_TARGET]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path, index=False)
    logger.info("Generated dummy data -> %s (%d rows)", output_path, len(train_df))

    return output_path


def train_model(
    df: pd.DataFrame,
    config: DatasetConfig,
    model_path: Optional[Path] = None,
    n_estimators: int = 200,
) -> dict:
    """
    Train a RandomForest model using the dataset configuration.

    Args:
        df: Training DataFrame with features and target
        config: Dataset configuration
        model_path: Override output model path (uses config default if None)
        n_estimators: Number of trees in the forest

    Returns:
        Dictionary with training metrics
    """
    feature_cols = config.feature_columns
    target_col = config.target_column

    # Validate columns exist
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    X = df[feature_cols]
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.train_test_split,
        random_state=config.random_state,
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    logger.info("Validation MAE: %.3f", mae)
    logger.info("Validation R^2: %.3f", r2)

    # Save model
    output_path = model_path or config.get_model_path()
    joblib.dump(model, output_path)
    logger.info("Saved model to %s", output_path.resolve())

    return {
        "mae": mae,
        "r2": r2,
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": str(output_path),
        "dataset": config.name,
    }


def process_and_train(
    dataset_name: str,
    registry: DatasetRegistry,
    model_path: Optional[Path] = None,
    generate: bool = False,
    dummy_rows: int = 600,
) -> dict:
    """
    Full pipeline: process data sources and train model.

    Args:
        dataset_name: Name of dataset from registry
        registry: Dataset registry
        model_path: Override model output path
        generate: Whether to (re)generate dummy data
        dummy_rows: Number of rows for dummy data generation

    Returns:
        Training metrics dictionary
    """
    config = registry.get(dataset_name)
    if config is None:
        available = registry.list_datasets()
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    loader = DatasetLoader(registry)

    # Handle dummy data generation
    if dataset_name == "dummy" and generate:
        generate_dummy_data(config.output_path, rows=dummy_rows)

    # Load and process the dataset
    logger.info("Loading dataset: %s", dataset_name)
    df = loader.load_dataset(dataset_name)
    logger.info("Loaded %d rows from %s", len(df), dataset_name)

    # Save processed data if different from source
    if config.sources:
        source_path = config.sources[0].path
        if config.output_path and source_path != config.output_path:
            loader.process_and_save(dataset_name)

    # Train the model
    return train_model(df, config, model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train RT prediction models using dataset configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --dataset tron
  python scripts/train.py --dataset dummy --generate
  python scripts/train.py --dataset tron --model-path models/custom.model
  python scripts/train.py --list-datasets
        """,
    )

    parser.add_argument(
        "--dataset",
        default="tron",
        help="Dataset name from datasets.yaml (default: tron)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override model output path",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate/regenerate dummy data before training",
    )
    parser.add_argument(
        "--dummy-rows",
        type=int,
        default=600,
        help="Number of rows for dummy data generation (default: 600)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to datasets.yaml (uses default if not specified)",
    )

    args = parser.parse_args()

    # Load registry
    if args.config:
        from core.datasets import load_datasets_from_yaml, _register_default_transformers
        registry = load_datasets_from_yaml(args.config)
        _register_default_transformers(registry)
    else:
        registry = get_dataset_registry()

    # List datasets mode
    if args.list_datasets:
        print("Available datasets:")
        for name in registry.list_datasets():
            config = registry.get(name)
            print(f"  {name}: {config.description}")
            print(f"    Sources: {[str(s.path) for s in config.sources]}")
            print(f"    Output: {config.output_path}")
        return

    # Train
    metrics = process_and_train(
        dataset_name=args.dataset,
        registry=registry,
        model_path=args.model_path,
        generate=args.generate,
        dummy_rows=args.dummy_rows,
    )

    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    main()
