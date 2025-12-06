"""
Base predictor interface for the Strategy Pattern.
All predictor implementations must inherit from BasePredictor.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BasePredictor(ABC):
    """
    Abstract base class for prediction strategies.

    Implementations can provide either:
    - predict_score(): Direct score prediction (regression)
    - predict_probabilities(): Probability distribution over buckets (classification)
    """

    def predict_score(self, features: Dict[str, float]) -> Optional[float]:
        """
        Predict the final RT score directly.

        Args:
            features: Dictionary of normalized feature values
                     (e.g., days_since_release, current_rating, num_reviews)

        Returns:
            Predicted score (0-100), or None if not supported
        """
        return None

    @abstractmethod
    def predict_probabilities(self, features: Dict[str, float]) -> Dict[int, float]:
        """
        Predict probability distribution over rating buckets.

        Args:
            features: Dictionary of normalized feature values
                     (e.g., days_since_release, current_rating, num_reviews)

        Returns:
            Dictionary mapping bucket ID (0, 1, 2) to probability (0.0-1.0)
            Probabilities should sum to 1.0
        """
        pass

    @property
    def is_regressor(self) -> bool:
        """Return True if this predictor supports direct score prediction."""
        return False

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Return a descriptive name for this predictor (e.g., 'dummy', 'prediction.model').
        Used for transparency in signal responses.
        """
        pass
