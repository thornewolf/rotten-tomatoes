"""
Compare XGBoost vs RandomForest models on RT prediction task.

Usage:
    python scripts/compare_models.py --dataset dummy
    python scripts/compare_models.py --dataset tron
    python scripts/compare_models.py --dataset dummy --generate
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor

from core.datasets import (
    DatasetConfig,
    DatasetLoader,
    DatasetRegistry,
    get_dataset_registry,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results from training a model."""
    name: str
    mae: float
    rmse: float
    r2: float
    cv_mae_mean: float
    cv_mae_std: float
    feature_importances: dict


def train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    random_state: int = 42,
) -> ModelResult:
    """Train and evaluate RandomForest model."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    cv_mae_mean = -cv_scores.mean()
    cv_mae_std = cv_scores.std()

    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_))

    return ModelResult(
        name="RandomForest",
        mae=mae,
        rmse=rmse,
        r2=r2,
        cv_mae_mean=cv_mae_mean,
        cv_mae_std=cv_mae_std,
        feature_importances=importances,
    )


def train_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    random_state: int = 42,
) -> ModelResult:
    """Train and evaluate XGBoost model."""
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    cv_mae_mean = -cv_scores.mean()
    cv_mae_std = cv_scores.std()

    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_))

    return ModelResult(
        name="XGBoost",
        mae=mae,
        rmse=rmse,
        r2=r2,
        cv_mae_mean=cv_mae_mean,
        cv_mae_std=cv_mae_std,
        feature_importances=importances,
    )


def compare_models(
    df: pd.DataFrame,
    config: DatasetConfig,
    save_models: bool = False,
    output_dir: Optional[Path] = None,
) -> tuple[ModelResult, ModelResult]:
    """
    Train and compare RandomForest vs XGBoost on the dataset.

    Returns:
        Tuple of (RandomForest result, XGBoost result)
    """
    feature_cols = config.feature_columns
    target_col = config.target_column

    # Validate columns
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_cols]
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.train_test_split,
        random_state=config.random_state,
    )

    logger.info(f"Dataset: {config.name}")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features: {feature_cols}")
    logger.info("")

    # Train models
    rf_result = train_random_forest(
        X_train, X_test, y_train, y_test,
        feature_cols, config.random_state
    )

    xgb_result = train_xgboost(
        X_train, X_test, y_train, y_test,
        feature_cols, config.random_state
    )

    return rf_result, xgb_result


def print_results(rf_result: ModelResult, xgb_result: ModelResult):
    """Pretty print comparison results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    # Table header
    print(f"\n{'Metric':<25} {'RandomForest':>15} {'XGBoost':>15} {'Winner':>10}")
    print("-" * 65)

    # MAE (lower is better)
    winner = "RF" if rf_result.mae < xgb_result.mae else "XGB"
    print(f"{'MAE (test)':<25} {rf_result.mae:>15.3f} {xgb_result.mae:>15.3f} {winner:>10}")

    # RMSE (lower is better)
    winner = "RF" if rf_result.rmse < xgb_result.rmse else "XGB"
    print(f"{'RMSE (test)':<25} {rf_result.rmse:>15.3f} {xgb_result.rmse:>15.3f} {winner:>10}")

    # R² (higher is better)
    winner = "RF" if rf_result.r2 > xgb_result.r2 else "XGB"
    print(f"{'R² (test)':<25} {rf_result.r2:>15.3f} {xgb_result.r2:>15.3f} {winner:>10}")

    # CV MAE
    winner = "RF" if rf_result.cv_mae_mean < xgb_result.cv_mae_mean else "XGB"
    rf_cv = f"{rf_result.cv_mae_mean:.3f}±{rf_result.cv_mae_std:.3f}"
    xgb_cv = f"{xgb_result.cv_mae_mean:.3f}±{xgb_result.cv_mae_std:.3f}"
    print(f"{'CV MAE (5-fold)':<25} {rf_cv:>15} {xgb_cv:>15} {winner:>10}")

    # Feature importances
    print("\n" + "-" * 65)
    print("FEATURE IMPORTANCES")
    print("-" * 65)
    print(f"{'Feature':<25} {'RandomForest':>15} {'XGBoost':>15}")
    print("-" * 55)

    for feature in rf_result.feature_importances:
        rf_imp = rf_result.feature_importances[feature]
        xgb_imp = xgb_result.feature_importances[feature]
        print(f"{feature:<25} {rf_imp:>15.3f} {xgb_imp:>15.3f}")

    # Summary
    print("\n" + "=" * 60)
    rf_wins = sum([
        rf_result.mae < xgb_result.mae,
        rf_result.rmse < xgb_result.rmse,
        rf_result.r2 > xgb_result.r2,
        rf_result.cv_mae_mean < xgb_result.cv_mae_mean,
    ])
    xgb_wins = 4 - rf_wins

    if rf_wins > xgb_wins:
        print(f"OVERALL WINNER: RandomForest ({rf_wins}/4 metrics)")
    elif xgb_wins > rf_wins:
        print(f"OVERALL WINNER: XGBoost ({xgb_wins}/4 metrics)")
    else:
        print("RESULT: TIE (2/4 metrics each)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare XGBoost vs RandomForest on RT prediction task.",
    )
    parser.add_argument(
        "--dataset",
        default="dummy",
        help="Dataset name from datasets.yaml (default: dummy)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate/regenerate dummy data before comparison",
    )
    parser.add_argument(
        "--dummy-rows",
        type=int,
        default=600,
        help="Number of rows for dummy data (default: 600)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    registry = get_dataset_registry()

    if args.list_datasets:
        print("Available datasets:")
        for name in registry.list_datasets():
            config = registry.get(name)
            print(f"  {name}: {config.description}")
        return

    # Get dataset config
    config = registry.get(args.dataset)
    if config is None:
        available = registry.list_datasets()
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {available}")

    loader = DatasetLoader(registry)

    # Generate dummy data if requested
    if args.dataset == "dummy" and args.generate:
        from scripts.train import generate_dummy_data
        generate_dummy_data(config.output_path, rows=args.dummy_rows)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    df = loader.load_dataset(args.dataset)
    logger.info(f"Loaded {len(df)} rows\n")

    # Compare models
    rf_result, xgb_result = compare_models(df, config)

    # Print results
    print_results(rf_result, xgb_result)


if __name__ == "__main__":
    main()
