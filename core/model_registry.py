"""
Model registry for managing ML model configurations.
Provides a centralized way to register, discover, and load prediction models.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default path for model registry configuration
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "models.yaml"


@dataclass
class ModelConfig:
    """Configuration for a single prediction model."""

    name: str
    path: Path
    description: str = ""
    is_default: bool = False
    model_type: str = "auto"  # auto, classifier, regressor

    def exists(self) -> bool:
        """Check if the model file exists."""
        return self.path.exists()


@dataclass
class ModelRegistry:
    """Registry for managing prediction models."""

    models: Dict[str, ModelConfig] = field(default_factory=dict)
    _default_model: Optional[str] = None

    def register(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self.models[config.name] = config
        if config.is_default:
            self._default_model = config.name
        logger.debug("Registered model: %s at %s", config.name, config.path)

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        return self.models.get(name)

    def get_default(self) -> Optional[ModelConfig]:
        """Get the default model configuration."""
        if self._default_model:
            return self.models.get(self._default_model)
        # Fallback to first available model
        if self.models:
            return next(iter(self.models.values()))
        return None

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())

    def list_available_models(self) -> List[str]:
        """List only models whose files exist."""
        return [name for name, config in self.models.items() if config.exists()]

    @property
    def default_name(self) -> Optional[str]:
        """Get the name of the default model."""
        return self._default_model


def load_registry_from_yaml(path: Path | str | None = None) -> ModelRegistry:
    """
    Load model registry from a YAML configuration file.

    Args:
        path: Path to the YAML file. Defaults to models.yaml in project root.

    Returns:
        Populated ModelRegistry instance.

    Example YAML format:
        models:
          default:
            path: prediction.model
            description: Production model trained on all data
            is_default: true
          tron:
            path: prediction_tron.model
            description: Model trained on Tron dataset
    """
    registry = ModelRegistry()
    config_path = Path(path) if path else DEFAULT_REGISTRY_PATH

    if not config_path.exists():
        logger.warning("Model registry file not found at %s", config_path)
        return registry

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        if not data or "models" not in data:
            logger.warning("No models defined in %s", config_path)
            return registry

        project_root = config_path.parent

        for name, config_data in data["models"].items():
            if not isinstance(config_data, dict):
                logger.warning("Invalid config for model %s, skipping", name)
                continue

            # Resolve path relative to config file location
            model_path = config_data.get("path", f"{name}.model")
            if not Path(model_path).is_absolute():
                model_path = project_root / model_path

            config = ModelConfig(
                name=name,
                path=Path(model_path),
                description=config_data.get("description", ""),
                is_default=config_data.get("is_default", False),
                model_type=config_data.get("model_type", "auto"),
            )
            registry.register(config)

        logger.info(
            "Loaded %d models from registry: %s",
            len(registry.models),
            registry.list_models(),
        )

    except yaml.YAMLError as exc:
        logger.error("Failed to parse model registry YAML: %s", exc)
    except OSError as exc:
        logger.error("Failed to read model registry file: %s", exc)

    return registry


def create_default_registry(project_root: Path | None = None) -> ModelRegistry:
    """
    Create a registry with default model configurations.
    Used as fallback when no YAML config exists.

    Args:
        project_root: Project root directory. Defaults to parent of this file's parent.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    registry = ModelRegistry()

    # Register default models
    default_models = [
        ModelConfig(
            name="default",
            path=project_root / "prediction.model",
            description="Production model",
            is_default=True,
        ),
        ModelConfig(
            name="tron",
            path=project_root / "prediction_tron.model",
            description="Model trained on Tron dataset",
        ),
        ModelConfig(
            name="dummy",
            path=project_root / "prediction_dummy.model",
            description="Model trained on synthetic data",
        ),
    ]

    for config in default_models:
        registry.register(config)

    return registry


# Global registry instance - lazily initialized
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    Loads from YAML if available, otherwise uses defaults.
    """
    global _registry_instance

    if _registry_instance is None:
        # Try loading from YAML first
        _registry_instance = load_registry_from_yaml()

        # If no models loaded, use defaults
        if not _registry_instance.models:
            logger.info("No YAML config found, using default model registry")
            _registry_instance = create_default_registry()

    return _registry_instance


def get_model_path(name: str | None = None) -> Path | None:
    """
    Convenience function to get a model path by name.

    Args:
        name: Model name. If None, returns the default model path.

    Returns:
        Path to the model file, or None if not found.
    """
    registry = get_model_registry()

    if name is None:
        config = registry.get_default()
    else:
        config = registry.get(name)

    return config.path if config else None
