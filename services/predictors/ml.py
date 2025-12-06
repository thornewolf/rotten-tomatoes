"""
Machine learning predictor using trained sklearn models.
Wraps a pickled classifier/regressor for production predictions.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from services.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


class MLPredictor(BasePredictor):
    """
    ML-based predictor using a trained sklearn model.

    Supports both:
    - Classifiers (using predict_proba for bucket probabilities)
    - Regressors (using predict for direct score prediction)
    """

    def __init__(self, model_path: Path | str):
        """
        Initialize the ML predictor.

        Args:
            model_path: Path to the pickled model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        self.model_path = Path(model_path)
        self.model: Any | None = None
        self._is_regressor: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the pickled model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.model = joblib.load(self.model_path)

            # Detect if this is a regressor (no predict_proba) or classifier
            self._is_regressor = not hasattr(self.model, "predict_proba")
            model_type = "regressor" if self._is_regressor else "classifier"
            logger.info("Loaded ML %s from %s", model_type, self.model_path)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", self.model_path, exc)
            raise

    @property
    def is_regressor(self) -> bool:
        """Return True if this is a regression model."""
        return self._is_regressor

    def predict_score(self, features: Dict[str, float]) -> Optional[float]:
        """
        Predict the final RT score directly (for regression models).

        Returns None if this is a classifier.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not self._is_regressor:
            return None

        # Build feature vector in correct order
        feature_cols = ["days_since_release", "current_rating", "num_reviews"]
        vector = np.array([[features[col] for col in feature_cols]])

        # Suppress sklearn feature name warning (we know the order is correct)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            pred = self.model.predict(vector)[0]

        # Clamp to valid range
        return max(0.0, min(100.0, float(pred)))

    def predict_probabilities(self, features: Dict[str, float]) -> Dict[int, float]:
        """
        Generate probability distribution using the ML model.

        For classifiers: uses predict_proba
        For regressors: converts predicted score to bucket probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Build feature vector in correct order
        feature_cols = ["days_since_release", "current_rating", "num_reviews"]
        vector = np.array([[features[col] for col in feature_cols]])

        # Suppress sklearn feature name warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")

            # For classifiers, use predict_proba
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(vector)[0]
                classes = getattr(self.model, "classes_", list(range(len(probs))))

                prob_map: Dict[int, float] = {}
                for idx, cls in enumerate(classes):
                    try:
                        cls_int = int(cls)
                    except (ValueError, TypeError):
                        continue
                    prob_map[cls_int] = float(probs[idx])

                # Normalize
                total = sum(prob_map.values())
                if total > 0:
                    prob_map = {k: v / total for k, v in prob_map.items()}

                return prob_map

            # For regressors, convert score to bucket
            pred = self.model.predict(vector)[0]

        score = max(0.0, min(100.0, float(pred)))

        # Map score to bucket (0: <60, 1: 60-90, 2: >90)
        if score < 60:
            bucket = 0
        elif score < 90:
            bucket = 1
        else:
            bucket = 2

        return {bucket: 1.0}

    @property
    def source_name(self) -> str:
        return self.model_path.name
