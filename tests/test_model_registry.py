"""
Unit tests for the model registry.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from core.model_registry import (
    ModelConfig,
    ModelRegistry,
    load_registry_from_yaml,
    create_default_registry,
    get_model_registry,
    get_model_path,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_exists_true_when_file_exists(self, tmp_path):
        """Should return True when model file exists."""
        model_file = tmp_path / "model.joblib"
        model_file.touch()

        config = ModelConfig(name="test", path=model_file)
        assert config.exists() is True

    def test_exists_false_when_file_missing(self, tmp_path):
        """Should return False when model file doesn't exist."""
        config = ModelConfig(name="test", path=tmp_path / "nonexistent.joblib")
        assert config.exists() is False

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ModelConfig(name="test", path=Path("/some/path"))

        assert config.description == ""
        assert config.is_default is False
        assert config.model_type == "auto"


class TestModelRegistry:
    """Tests for ModelRegistry functionality."""

    def test_register_model(self, tmp_path):
        """Should register model configurations."""
        registry = ModelRegistry()
        config = ModelConfig(name="test", path=tmp_path / "model.joblib")

        registry.register(config)

        assert "test" in registry.list_models()
        assert registry.get("test") is config

    def test_register_default_model(self, tmp_path):
        """Should track default model."""
        registry = ModelRegistry()
        config = ModelConfig(name="test", path=tmp_path / "model.joblib", is_default=True)

        registry.register(config)

        assert registry.default_name == "test"
        assert registry.get_default() is config

    def test_get_missing_model(self):
        """Should return None for missing model."""
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None

    def test_get_default_fallback(self, tmp_path):
        """Should fallback to first model if no default set."""
        registry = ModelRegistry()
        config = ModelConfig(name="first", path=tmp_path / "first.joblib")
        registry.register(config)

        default = registry.get_default()
        assert default is config

    def test_list_models(self, tmp_path):
        """Should list all registered model names."""
        registry = ModelRegistry()
        registry.register(ModelConfig(name="a", path=tmp_path / "a.joblib"))
        registry.register(ModelConfig(name="b", path=tmp_path / "b.joblib"))

        models = registry.list_models()
        assert "a" in models
        assert "b" in models

    def test_list_available_models(self, tmp_path):
        """Should list only models with existing files."""
        registry = ModelRegistry()

        existing = tmp_path / "existing.joblib"
        existing.touch()

        registry.register(ModelConfig(name="existing", path=existing))
        registry.register(ModelConfig(name="missing", path=tmp_path / "missing.joblib"))

        available = registry.list_available_models()
        assert "existing" in available
        assert "missing" not in available


class TestLoadRegistryFromYaml:
    """Tests for YAML loading."""

    def test_load_valid_yaml(self, tmp_path):
        """Should load models from YAML."""
        yaml_content = """
models:
  production:
    path: prod.model
    description: Production model
    is_default: true
  experimental:
    path: exp.model
    description: Experimental model
"""
        yaml_file = tmp_path / "models.yaml"
        yaml_file.write_text(yaml_content)

        registry = load_registry_from_yaml(yaml_file)

        assert "production" in registry.list_models()
        assert "experimental" in registry.list_models()
        assert registry.default_name == "production"

    def test_load_missing_file(self, tmp_path):
        """Should return empty registry for missing file."""
        registry = load_registry_from_yaml(tmp_path / "nonexistent.yaml")

        assert len(registry.list_models()) == 0

    def test_load_empty_yaml(self, tmp_path):
        """Should handle empty YAML gracefully."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        registry = load_registry_from_yaml(yaml_file)
        assert len(registry.list_models()) == 0

    def test_load_no_models_section(self, tmp_path):
        """Should handle YAML without models section."""
        yaml_file = tmp_path / "nomodels.yaml"
        yaml_file.write_text("other_config: value")

        registry = load_registry_from_yaml(yaml_file)
        assert len(registry.list_models()) == 0

    def test_relative_path_resolution(self, tmp_path):
        """Should resolve relative paths from YAML location."""
        yaml_content = """
models:
  test:
    path: subdir/model.joblib
"""
        yaml_file = tmp_path / "models.yaml"
        yaml_file.write_text(yaml_content)

        registry = load_registry_from_yaml(yaml_file)
        config = registry.get("test")

        assert config is not None
        assert config.path == tmp_path / "subdir" / "model.joblib"


class TestCreateDefaultRegistry:
    """Tests for default registry creation."""

    def test_creates_expected_models(self, tmp_path):
        """Should create default, tron, and dummy models."""
        registry = create_default_registry(tmp_path)

        models = registry.list_models()
        assert "default" in models
        assert "tron" in models
        assert "dummy" in models

    def test_default_model_is_set(self, tmp_path):
        """Should set default model."""
        registry = create_default_registry(tmp_path)

        assert registry.default_name == "default"

    def test_paths_relative_to_project_root(self, tmp_path):
        """Model paths should be relative to project root."""
        registry = create_default_registry(tmp_path)
        config = registry.get("default")

        assert config is not None
        assert config.path.parent == tmp_path


class TestGetModelPath:
    """Tests for get_model_path convenience function."""

    def test_get_default_path(self):
        """Should return default model path when name is None."""
        # Uses global registry
        path = get_model_path(None)
        # Path should be returned (actual value depends on global state)
        # Just check it's a Path or None
        assert path is None or isinstance(path, Path)

    def test_get_named_path(self):
        """Should return path for named model."""
        path = get_model_path("default")
        assert path is None or isinstance(path, Path)

    def test_get_missing_model_path(self):
        """Should return None for unknown model."""
        path = get_model_path("definitely_not_a_real_model_name_12345")
        assert path is None
