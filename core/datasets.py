"""
Dataset configuration and management system.

Provides a unified interface for defining, loading, and processing datasets
for model training. Supports multiple data sources, transformations, and
configurations through YAML files.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Standard feature columns expected by the model
STANDARD_FEATURES = ["days_since_release", "current_rating", "num_reviews"]
STANDARD_TARGET = "final_score"


@dataclass
class DataSource:
    """
    A single data source (raw CSV file).

    Attributes:
        path: Path to the raw data file
        format: Data format type (e.g., "reviews", "snapshots", "processed")
        transform: Optional transformation to apply (e.g., "review_prefix")
        transform_args: Arguments for the transformation
    """

    path: Path
    format: str = "processed"  # "reviews", "snapshots", "processed"
    transform: Optional[str] = None
    transform_args: Dict[str, Any] = field(default_factory=dict)

    def exists(self) -> bool:
        """Check if the source file exists."""
        return self.path.exists()


@dataclass
class DatasetConfig:
    """
    Configuration for a complete dataset.

    A dataset can combine multiple sources and define how they should be
    processed and merged for training.

    Attributes:
        name: Unique identifier for this dataset
        description: Human-readable description
        sources: List of data sources to combine
        output_path: Where to save the processed dataset
        feature_columns: Feature columns to use (defaults to standard)
        target_column: Target column name
        train_test_split: Fraction to use for testing (0-1)
        random_state: Random seed for reproducibility
    """

    name: str
    description: str = ""
    sources: List[DataSource] = field(default_factory=list)
    output_path: Optional[Path] = None
    feature_columns: List[str] = field(default_factory=lambda: STANDARD_FEATURES.copy())
    target_column: str = STANDARD_TARGET
    train_test_split: float = 0.2
    random_state: int = 42

    def __post_init__(self):
        # Set default output path if not specified
        if self.output_path is None:
            self.output_path = Path(f"data/processed_{self.name}.csv")

    def get_model_path(self) -> Path:
        """Get the default model output path for this dataset."""
        return Path(f"prediction_{self.name}.model")


class DatasetRegistry:
    """
    Registry for managing dataset configurations.

    Loads configurations from YAML and provides access to datasets by name.
    """

    def __init__(self):
        self.datasets: Dict[str, DatasetConfig] = {}
        self._transformers: Dict[str, Callable] = {}

    def register_transformer(self, name: str, func: Callable) -> None:
        """Register a data transformation function."""
        self._transformers[name] = func

    def get_transformer(self, name: str) -> Optional[Callable]:
        """Get a registered transformer by name."""
        return self._transformers.get(name)

    def register(self, config: DatasetConfig) -> None:
        """Register a dataset configuration."""
        self.datasets[config.name] = config
        logger.debug("Registered dataset: %s", config.name)

    def get(self, name: str) -> Optional[DatasetConfig]:
        """Get a dataset configuration by name."""
        return self.datasets.get(name)

    def list_datasets(self) -> List[str]:
        """List all registered dataset names."""
        return list(self.datasets.keys())


def load_datasets_from_yaml(
    path: Path | str,
    base_dir: Optional[Path] = None
) -> DatasetRegistry:
    """
    Load dataset configurations from a YAML file.

    Args:
        path: Path to the YAML configuration file
        base_dir: Base directory for resolving relative paths (defaults to YAML dir)

    Returns:
        DatasetRegistry populated with configurations

    Example YAML format:
        datasets:
          tron:
            description: "Tron movie review data"
            sources:
              - path: data/tron-sample.csv
                format: reviews
                transform: review_prefix
                transform_args:
                  prefix_lengths: [3, 5, 10, 20, 30, 40, 50]
            output_path: data/processed_tron.csv

          dummy:
            description: "Synthetic training data"
            sources:
              - path: data/processed_dummy.csv
                format: processed
    """
    registry = DatasetRegistry()
    config_path = Path(path)

    if not config_path.exists():
        logger.warning("Dataset config file not found: %s", config_path)
        return registry

    if base_dir is None:
        base_dir = config_path.parent

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        if not data or "datasets" not in data:
            logger.warning("No datasets defined in %s", config_path)
            return registry

        for name, config_data in data["datasets"].items():
            sources = []
            for src_data in config_data.get("sources", []):
                src_path = Path(src_data["path"])
                if not src_path.is_absolute():
                    src_path = base_dir / src_path

                sources.append(DataSource(
                    path=src_path,
                    format=src_data.get("format", "processed"),
                    transform=src_data.get("transform"),
                    transform_args=src_data.get("transform_args", {}),
                ))

            output_path = config_data.get("output_path")
            if output_path:
                output_path = Path(output_path)
                if not output_path.is_absolute():
                    output_path = base_dir / output_path

            config = DatasetConfig(
                name=name,
                description=config_data.get("description", ""),
                sources=sources,
                output_path=output_path,
                feature_columns=config_data.get("feature_columns", STANDARD_FEATURES.copy()),
                target_column=config_data.get("target_column", STANDARD_TARGET),
                train_test_split=config_data.get("train_test_split", 0.2),
                random_state=config_data.get("random_state", 42),
            )
            registry.register(config)

        logger.info("Loaded %d datasets from %s", len(registry.datasets), config_path)

    except yaml.YAMLError as exc:
        logger.error("Failed to parse dataset config: %s", exc)
    except OSError as exc:
        logger.error("Failed to read dataset config: %s", exc)

    return registry


class DatasetLoader:
    """
    Loads and processes datasets based on configuration.

    Handles:
    - Loading raw data from various formats
    - Applying transformations
    - Combining multiple sources
    - Saving processed output
    """

    def __init__(self, registry: DatasetRegistry):
        self.registry = registry

    def load_source(self, source: DataSource) -> pd.DataFrame:
        """Load a single data source."""
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source.path}")

        df = pd.read_csv(source.path, encoding="utf-8-sig")
        logger.info("Loaded %d rows from %s", len(df), source.path)

        # Apply transformation if specified
        if source.transform:
            transformer = self.registry.get_transformer(source.transform)
            if transformer is None:
                raise ValueError(f"Unknown transformer: {source.transform}")
            df = transformer(df, **source.transform_args)
            logger.info("Applied transform '%s': %d rows", source.transform, len(df))

        return df

    def load_dataset(self, name: str) -> pd.DataFrame:
        """
        Load and combine all sources for a dataset.

        Args:
            name: Dataset name from registry

        Returns:
            Combined and processed DataFrame
        """
        config = self.registry.get(name)
        if config is None:
            raise ValueError(f"Unknown dataset: {name}")

        if not config.sources:
            raise ValueError(f"Dataset '{name}' has no sources defined")

        # Load and combine all sources
        dfs = []
        for source in config.sources:
            df = self.load_source(source)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        logger.info("Combined %d sources: %d total rows", len(dfs), len(combined))

        # Validate required columns
        required = config.feature_columns + [config.target_column]
        missing = [col for col in required if col not in combined.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return combined

    def process_and_save(self, name: str) -> Path:
        """
        Load, process, and save a dataset.

        Args:
            name: Dataset name from registry

        Returns:
            Path to the saved processed file
        """
        config = self.registry.get(name)
        if config is None:
            raise ValueError(f"Unknown dataset: {name}")

        df = self.load_dataset(name)

        # Select only the columns needed for training
        columns = config.feature_columns + [config.target_column]
        df = df[columns]

        # Save to output path
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.output_path, index=False)
        logger.info("Saved processed dataset to %s (%d rows)", config.output_path, len(df))

        return config.output_path


# Default config path
DEFAULT_DATASETS_PATH = Path(__file__).resolve().parent.parent / "datasets.yaml"

# Global registry instance
_registry_instance: Optional[DatasetRegistry] = None


def get_dataset_registry() -> DatasetRegistry:
    """Get the global dataset registry, loading from YAML if needed."""
    global _registry_instance

    if _registry_instance is None:
        _registry_instance = load_datasets_from_yaml(DEFAULT_DATASETS_PATH)
        _register_default_transformers(_registry_instance)

    return _registry_instance


def _register_default_transformers(registry: DatasetRegistry) -> None:
    """Register built-in data transformers."""
    from services.data_processing import ReviewProcessor, ColumnMapping

    def review_prefix_transform(
        df: pd.DataFrame,
        prefix_lengths: List[int] = None,
        column_mapping: Dict[str, str] = None,
    ) -> pd.DataFrame:
        """Transform review data to prefix features."""
        if prefix_lengths is None:
            prefix_lengths = [3, 5, 10, 20, 30, 40, 50]

        mapping = ColumnMapping(**(column_mapping or {}))
        processor = ReviewProcessor(column_mapping=mapping)

        # Validate and process
        validated_df = processor.validate_and_load_df(df)
        rows = processor.build_prefix_features(validated_df, prefix_lengths)

        return pd.DataFrame(rows)

    # Add validate_and_load_df method to ReviewProcessor if needed
    registry.register_transformer("review_prefix", review_prefix_transform)
