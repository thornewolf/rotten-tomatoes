import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.schemas import MarketSummary, PredictionResult, SignalResponse, RTContext
from services.predictors import BasePredictor, DummyPredictor, MLPredictor

logger = logging.getLogger(__name__)

# Feature columns expected by the trained model.
FEATURE_COLUMNS = ["days_since_release", "current_rating", "num_reviews"]

# Conservative defaults if the client omits a feature.
DEFAULT_FEATURE_VALUES = {
    "days_since_release": 30.0,
    "current_rating": 75.0,
    "num_reviews": 50.0,
}

# Map rating buckets to an approximate Rotten Tomatoes percentage for explainability.
BUCKET_TO_SCORE = {0: 62.0, 1: 78.0, 2: 92.0}


class PredictionEngine:
    """
    Orchestrates predictions using the Strategy Pattern.
    Automatically falls back to DummyPredictor if ML model is unavailable.
    """

    def __init__(self, model_path: Path | str = Path("models/prediction.model")):
        self.model_path = Path(model_path)
        self.predictor: BasePredictor = self._initialize_predictor()

    def _initialize_predictor(self) -> BasePredictor:
        """
        Try to load ML predictor, fall back to dummy if unavailable.
        """
        if not self.model_path.exists():
            logger.warning(
                "Prediction model not found at %s; using dummy predictor instead.",
                self.model_path,
            )
            return DummyPredictor()

        try:
            predictor = MLPredictor(self.model_path)
            logger.info("Loaded ML predictor from %s", self.model_path)
            return predictor
        except FileNotFoundError as exc:
            logger.warning(
                "Model file not found at %s: %s. Falling back to dummy predictor.",
                self.model_path,
                exc,
            )
            return DummyPredictor()
        except (OSError, IOError) as exc:
            logger.warning(
                "I/O error loading model at %s: %s. Falling back to dummy predictor.",
                self.model_path,
                exc,
            )
            return DummyPredictor()
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning(
                "Invalid model format at %s: %s. Falling back to dummy predictor.",
                self.model_path,
                exc,
            )
            return DummyPredictor()

    def _build_feature_vector(
        self, features: Dict[str, Any]
    ) -> tuple[list[float], Dict[str, float], list[str]]:
        normalized: Dict[str, float] = {}
        defaults_used: list[str] = []
        for key, default in DEFAULT_FEATURE_VALUES.items():
            raw_val = features.get(key)
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                val = default
                defaults_used.append(key)
            normalized[key] = val
        vector = [normalized[col] for col in FEATURE_COLUMNS]
        return vector, normalized, defaults_used


    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Generate a prediction using the configured predictor strategy.
        """
        _, normalized_features, defaults_used = self._build_feature_vector(features)

        # Try direct score prediction first (for regression models)
        direct_score = self.predictor.predict_score(normalized_features)

        if direct_score is not None:
            # Regression model - use predicted score directly
            expected_score = direct_score

            # Derive bucket from score
            if expected_score < 60:
                predicted_bucket = 0
            elif expected_score < 90:
                predicted_bucket = 1
            else:
                predicted_bucket = 2

            # For regressors, confidence is based on how far from bucket boundaries
            # (closer to boundaries = less confident)
            bucket_centers = {0: 30, 1: 75, 2: 95}
            distance_to_center = abs(expected_score - bucket_centers[predicted_bucket])
            max_distance = 30  # Max distance within a bucket
            confidence = max(0.5, 1.0 - (distance_to_center / max_distance) * 0.5)

            prob_map = {predicted_bucket: 1.0}
        else:
            # Classification model - use bucket probabilities
            prob_map = self.predictor.predict_probabilities(normalized_features)

            # Ensure the map always contains the expected buckets
            for bucket in BUCKET_TO_SCORE:
                prob_map.setdefault(bucket, 0.0)

            # Normalize probabilities
            total = sum(prob_map.values())
            if total > 0:
                prob_map = {k: v / total for k, v in prob_map.items()}

            # Calculate derived metrics
            predicted_bucket = max(prob_map, key=prob_map.get)
            confidence = prob_map.get(predicted_bucket, 0.0)
            expected_score = sum(
                BUCKET_TO_SCORE.get(cls, 75.0) * prob for cls, prob in prob_map.items()
            )

        fair_yes_price = max(0.0, min(prob_map.get(2, confidence), 1.0))
        fair_no_price = round(1 - fair_yes_price, 4)

        return PredictionResult(
            predicted_bucket=predicted_bucket,
            predicted_final_score=round(expected_score, 2),
            confidence=round(confidence, 4),
            probabilities={str(k): round(v, 4) for k, v in prob_map.items()},
            fair_yes_price=round(fair_yes_price, 4),
            fair_no_price=fair_no_price,
            features=normalized_features,
            defaults_used=defaults_used,
            model_source=self.predictor.source_name,
        )

    def generate_signal_payload(
        self, ticker: str, features: Dict[str, Any], market_snapshot: MarketSummary | None
    ) -> SignalResponse:
        prediction = self.predict(features)

        yes_bid = market_snapshot.yes_bid if market_snapshot else None
        yes_ask = market_snapshot.yes_ask if market_snapshot else None

        market_mid: float | None = None
        if yes_bid is not None and yes_ask is not None:
            market_mid = round((float(yes_bid) + float(yes_ask)) / 2, 4)
        elif yes_bid is not None:
            market_mid = float(yes_bid)
        elif yes_ask is not None:
            market_mid = float(yes_ask)

        delta_to_market: float | None = None
        if market_mid is not None:
            delta_to_market = round(prediction.fair_yes_price - market_mid, 4)

        return SignalResponse(
            ticker=ticker,
            predicted_final_score=prediction.predicted_final_score,
            confidence=prediction.confidence,
            fair_yes_price=prediction.fair_yes_price,
            fair_no_price=prediction.fair_no_price,
            delta_to_market=delta_to_market,
            probabilities=prediction.probabilities,
            model_source=prediction.model_source,
            rt_context=RTContext(
                features_used=prediction.features,
                defaults_used=prediction.defaults_used,
                predicted_bucket=prediction.predicted_bucket,
            ),
            market_context={
                "market_yes_mid_price": market_mid,
                "market_summary": market_snapshot.model_dump() if market_snapshot else {},
            },
        )


# Default model path for dependency injection
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "prediction.model"

# Singleton instance - will be lazily initialized
_engine_instance: PredictionEngine | None = None


def get_prediction_engine(model_path: Path | str | None = None) -> PredictionEngine:
    """
    Get or create the singleton PredictionEngine instance.
    Uses lazy initialization to avoid loading the model at import time.
    """
    global _engine_instance

    if _engine_instance is None:
        path = model_path or DEFAULT_MODEL_PATH
        _engine_instance = PredictionEngine(model_path=path)
        logger.info("Initialized PredictionEngine with model path: %s", path)

    return _engine_instance
