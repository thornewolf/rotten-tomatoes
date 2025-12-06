"""
Heuristic/dummy predictor based on simple rules.
Used as fallback when ML model is unavailable or for testing.
"""

import logging
from typing import Dict

from services.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


class DummyPredictor(BasePredictor):
    """
    Rule-based predictor using simple heuristics.

    Estimates probabilities based primarily on current_rating,
    with adjustments for the other features.
    """

    def predict_probabilities(self, features: Dict[str, float]) -> Dict[int, float]:
        """
        Generate probability distribution using heuristic rules.

        The logic scales probabilities based on current_rating:
        - Higher ratings increase prob of bucket 2 (high)
        - Middle ratings favor bucket 1 (mid)
        - Lower ratings increase prob of bucket 0 (low)
        """
        current_rating = features.get("current_rating", 75.0)
        scaled = max(0.0, min(current_rating / 100.0, 1.0))

        prob_high = min(0.15 + scaled * 0.55, 0.9)
        prob_mid = min(0.2 + (0.5 - abs(0.5 - scaled)), 0.75)
        prob_low = max(0.0, 1.0 - prob_high - prob_mid)

        total = prob_low + prob_mid + prob_high
        if total <= 0:
            # Fallback to uniform distribution
            return {0: 0.34, 1: 0.33, 2: 0.33}

        return {
            0: prob_low / total,
            1: prob_mid / total,
            2: prob_high / total,
        }

    @property
    def source_name(self) -> str:
        return "dummy"
