import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

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
    Wrapper around the pickled RandomForest model with graceful fallbacks.
    If prediction.model is missing at import time we keep serving dummy signals.
    """

    def __init__(self, model_path: Path | str = Path("prediction.model")):
        self.model_path = Path(model_path)
        self.model: Any | None = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.warning(
                "Prediction model not found at %s; serving dummy signals instead.",
                self.model_path,
            )
            return
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_loaded = True
            logger.info("Loaded prediction model from %s", self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load prediction model at %s: %s. Falling back to dummy signals.",
                self.model_path,
                exc,
            )
            self.model = None
            self.model_loaded = False

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

    def _dummy_probabilities(self, normalized_features: Dict[str, float]) -> Dict[int, float]:
        current_rating = normalized_features.get(
            "current_rating", DEFAULT_FEATURE_VALUES["current_rating"]
        )
        scaled = max(0.0, min(current_rating / 100.0, 1.0))

        prob_high = min(0.15 + scaled * 0.55, 0.9)
        prob_mid = min(0.2 + (0.5 - abs(0.5 - scaled)), 0.75)
        prob_low = max(0.0, 1.0 - prob_high - prob_mid)

        total = prob_low + prob_mid + prob_high
        if total <= 0:
            return {0: 0.34, 1: 0.33, 2: 0.33}
        return {
            0: prob_low / total,
            1: prob_mid / total,
            2: prob_high / total,
        }

    def _probabilities_from_model(self, vector: list[float]) -> Optional[Dict[int, float]]:
        if not self.model_loaded or self.model is None:
            return None

        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba([vector])[0]
                classes = getattr(self.model, "classes_", list(range(len(probs))))
                prob_map: Dict[int, float] = {}
                for idx, cls in enumerate(classes):
                    try:
                        cls_int = int(cls)
                    except Exception:
                        continue
                    prob_map[cls_int] = float(probs[idx])

                total = sum(prob_map.values())
                if total > 0:
                    prob_map = {k: v / total for k, v in prob_map.items()}
                return prob_map

            # Classifiers without predict_proba.
            pred = self.model.predict([vector])[0]
            try:
                label = int(pred)
            except Exception:
                label = 1
            return {label: 1.0}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prediction failed; using dummy probabilities: %s", exc)
            return None

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        vector, normalized_features, defaults_used = self._build_feature_vector(features)
        prob_map = self._probabilities_from_model(vector)

        model_source = (
            self.model_path.name if prob_map is not None and self.model_loaded else "dummy"
        )
        if prob_map is None:
            prob_map = self._dummy_probabilities(normalized_features)

        # Ensure the map always contains the expected buckets.
        for bucket in BUCKET_TO_SCORE:
            prob_map.setdefault(bucket, 0.0)

        total = sum(prob_map.values())
        if total > 0:
            prob_map = {k: v / total for k, v in prob_map.items()}

        predicted_bucket = max(prob_map, key=prob_map.get)
        confidence = prob_map.get(predicted_bucket, 0.0)
        expected_score = sum(
            BUCKET_TO_SCORE.get(cls, 75.0) * prob for cls, prob in prob_map.items()
        )

        fair_yes_price = max(0.0, min(prob_map.get(2, confidence), 1.0))
        fair_no_price = round(1 - fair_yes_price, 4)

        return {
            "predicted_bucket": predicted_bucket,
            "predicted_final_score": round(expected_score, 2),
            "confidence": round(confidence, 4),
            "probabilities": {str(k): round(v, 4) for k, v in prob_map.items()},
            "fair_yes_price": round(fair_yes_price, 4),
            "fair_no_price": fair_no_price,
            "features": normalized_features,
            "defaults_used": defaults_used,
            "model_source": model_source,
        }

    def generate_signal_payload(
        self, ticker: str, features: Dict[str, Any], market_snapshot: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        prediction = self.predict(features)
        market_snapshot = market_snapshot or {}

        top_of_book = market_snapshot.get("top_of_book", {}) or {}
        yes_top = top_of_book.get("yes", {}) or {}
        yes_bid = yes_top.get("bid")
        yes_ask = yes_top.get("ask")

        market_mid: float | None = None
        if yes_bid is not None and yes_ask is not None:
            market_mid = round((float(yes_bid) + float(yes_ask)) / 2, 4)
        elif yes_bid is not None:
            market_mid = float(yes_bid)
        elif yes_ask is not None:
            market_mid = float(yes_ask)

        delta_to_market: float | None = None
        if market_mid is not None:
            delta_to_market = round(prediction["fair_yes_price"] - market_mid, 4)

        return {
            "ticker": ticker,
            "predicted_final_score": prediction["predicted_final_score"],
            "confidence": prediction["confidence"],
            "fair_yes_price": prediction["fair_yes_price"],
            "fair_no_price": prediction["fair_no_price"],
            "delta_to_market": delta_to_market,
            "probabilities": prediction["probabilities"],
            "model_source": prediction["model_source"],
            "rt_context": {
                "features_used": prediction["features"],
                "defaults_used": prediction["defaults_used"],
                "predicted_bucket": prediction["predicted_bucket"],
            },
            "market_context": {
                "top_of_book": top_of_book,
                "orderbook": market_snapshot.get("orderbook", {}),
                "market_yes_mid_price": market_mid,
                "market_summary": {
                    "title": market_snapshot.get("title"),
                    "status": market_snapshot.get("status"),
                    "close_time": market_snapshot.get("close_time"),
                    "series_ticker": market_snapshot.get("series_ticker"),
                    "event_ticker": market_snapshot.get("event_ticker"),
                    "volume": market_snapshot.get("volume"),
                    "open_interest": market_snapshot.get("open_interest"),
                    "yes_bid": market_snapshot.get("yes_bid"),
                    "yes_ask": market_snapshot.get("yes_ask"),
                    "no_bid": market_snapshot.get("no_bid"),
                    "no_ask": market_snapshot.get("no_ask"),
                },
            },
        }


# Initialize at import so we can log if the model is missing.
_default_model_path = Path(__file__).resolve().parent.parent / "prediction.model"
prediction_engine = PredictionEngine(model_path=_default_model_path)
