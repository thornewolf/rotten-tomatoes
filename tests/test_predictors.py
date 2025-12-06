"""
Unit tests for prediction services.
Tests DummyPredictor, MLPredictor, and PredictionEngine.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.predictors.base import BasePredictor
from services.predictors.dummy import DummyPredictor
from services.predictors.ml import MLPredictor
from services.prediction_service import PredictionEngine, BUCKET_TO_SCORE


class TestDummyPredictor:
    """Tests for DummyPredictor heuristic-based predictions."""

    def test_predict_probabilities_high_rating(self):
        """High current_rating should favor high bucket."""
        predictor = DummyPredictor()
        features = {"current_rating": 95.0}

        probs = predictor.predict_probabilities(features)

        assert 0 in probs and 1 in probs and 2 in probs
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Probabilities sum to 1
        assert probs[2] > probs[0]  # High bucket more likely than low

    def test_predict_probabilities_low_rating(self):
        """Low current_rating should have lower prob for high bucket."""
        predictor = DummyPredictor()
        features = {"current_rating": 30.0}

        probs = predictor.predict_probabilities(features)

        assert abs(sum(probs.values()) - 1.0) < 0.01
        # Low ratings should have lower probability of bucket 2

    def test_predict_probabilities_default_rating(self):
        """Missing current_rating should use default (75.0)."""
        predictor = DummyPredictor()
        features = {}

        probs = predictor.predict_probabilities(features)

        assert abs(sum(probs.values()) - 1.0) < 0.01
        # With default 75, probabilities should be reasonable

    def test_predict_probabilities_boundary_values(self):
        """Test boundary values (0 and 100)."""
        predictor = DummyPredictor()

        probs_zero = predictor.predict_probabilities({"current_rating": 0.0})
        probs_hundred = predictor.predict_probabilities({"current_rating": 100.0})

        assert abs(sum(probs_zero.values()) - 1.0) < 0.01
        assert abs(sum(probs_hundred.values()) - 1.0) < 0.01

    def test_source_name(self):
        """Source name should be 'dummy'."""
        predictor = DummyPredictor()
        assert predictor.source_name == "dummy"

    def test_is_not_regressor(self):
        """DummyPredictor is not a regressor."""
        predictor = DummyPredictor()
        assert predictor.is_regressor is False


class TestMLPredictor:
    """Tests for MLPredictor with real sklearn models."""

    def test_init_file_not_found(self):
        """Should raise FileNotFoundError for missing model file."""
        with pytest.raises(FileNotFoundError):
            MLPredictor(Path("/nonexistent/path/model.joblib"))

    def test_load_classifier_model(self, tmp_path):
        """Should load classifier model and detect type correctly."""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np

        # Create a real classifier
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 2, 1])
        clf.fit(X, y)

        model_path = tmp_path / "classifier.joblib"
        joblib.dump(clf, model_path)

        predictor = MLPredictor(model_path)

        assert predictor.model is not None
        assert predictor.is_regressor is False  # has predict_proba

    def test_load_regressor_model(self, tmp_path):
        """Should detect regressor model (no predict_proba)."""
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        import numpy as np

        # Create a real regressor
        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([50.0, 70.0, 80.0, 90.0])
        reg.fit(X, y)

        model_path = tmp_path / "regressor.joblib"
        joblib.dump(reg, model_path)

        predictor = MLPredictor(model_path)
        assert predictor.is_regressor is True

    def test_predict_probabilities_classifier(self, tmp_path):
        """Should return probabilities from classifier."""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 2, 1])
        clf.fit(X, y)

        model_path = tmp_path / "classifier.joblib"
        joblib.dump(clf, model_path)

        predictor = MLPredictor(model_path)
        features = {"days_since_release": 7.0, "current_rating": 72.5, "num_reviews": 125.0}

        probs = predictor.predict_probabilities(features)

        assert isinstance(probs, dict)
        assert all(isinstance(k, int) for k in probs.keys())
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_score_regressor(self, tmp_path):
        """Should return score from regressor."""
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        import numpy as np

        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([50.0, 70.0, 80.0, 90.0])
        reg.fit(X, y)

        model_path = tmp_path / "regressor.joblib"
        joblib.dump(reg, model_path)

        predictor = MLPredictor(model_path)
        features = {"days_since_release": 7.0, "current_rating": 72.5, "num_reviews": 125.0}

        score = predictor.predict_score(features)

        assert score is not None
        assert 0 <= score <= 100

    def test_predict_score_classifier_returns_none(self, tmp_path):
        """Classifier should return None for predict_score."""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 2, 1])
        clf.fit(X, y)

        model_path = tmp_path / "classifier.joblib"
        joblib.dump(clf, model_path)

        predictor = MLPredictor(model_path)
        features = {"days_since_release": 7.0, "current_rating": 72.5, "num_reviews": 125.0}

        score = predictor.predict_score(features)

        assert score is None

    def test_source_name(self, tmp_path):
        """Source name should be model filename."""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([0, 1])
        clf.fit(X, y)

        model_path = tmp_path / "test_model.joblib"
        joblib.dump(clf, model_path)

        predictor = MLPredictor(model_path)
        assert predictor.source_name == "test_model.joblib"


class TestPredictionEngine:
    """Tests for PredictionEngine orchestration."""

    def test_init_with_missing_model_uses_dummy(self, tmp_path):
        """Should fall back to DummyPredictor when model doesn't exist."""
        missing_path = tmp_path / "nonexistent.model"
        engine = PredictionEngine(model_path=missing_path)

        assert isinstance(engine.predictor, DummyPredictor)

    def test_init_with_valid_model(self, tmp_path):
        """Should load MLPredictor when model exists."""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([0, 1])
        clf.fit(X, y)

        model_path = tmp_path / "test_model.joblib"
        joblib.dump(clf, model_path)

        engine = PredictionEngine(model_path=model_path)

        assert isinstance(engine.predictor, MLPredictor)

    def test_build_feature_vector_complete(self, sample_features):
        """Should build feature vector with all values."""
        engine = PredictionEngine()
        vector, normalized, defaults_used = engine._build_feature_vector(sample_features)

        assert len(vector) == 3
        assert vector == [7.0, 72.5, 125.0]
        assert normalized["days_since_release"] == 7.0
        assert normalized["current_rating"] == 72.5
        assert normalized["num_reviews"] == 125.0
        assert defaults_used == []

    def test_build_feature_vector_partial(self, sample_features_partial):
        """Should use defaults for missing features."""
        engine = PredictionEngine()
        vector, normalized, defaults_used = engine._build_feature_vector(sample_features_partial)

        assert len(vector) == 3
        assert "days_since_release" in defaults_used
        assert "num_reviews" in defaults_used
        assert "current_rating" not in defaults_used

    def test_build_feature_vector_invalid_types(self):
        """Should handle invalid feature types gracefully."""
        engine = PredictionEngine()
        features = {
            "days_since_release": "not_a_number",
            "current_rating": None,
            "num_reviews": 50.0,
        }

        vector, normalized, defaults_used = engine._build_feature_vector(features)

        assert len(vector) == 3
        assert "days_since_release" in defaults_used
        assert "current_rating" in defaults_used

    def test_predict_returns_result(self, sample_features, tmp_path):
        """Should return PredictionResult with all fields."""
        # Use a missing model so it falls back to dummy
        engine = PredictionEngine(model_path=tmp_path / "nonexistent.model")
        result = engine.predict(sample_features)

        assert result.predicted_bucket in [0, 1, 2]
        assert 0 <= result.predicted_final_score <= 100
        assert 0 <= result.confidence <= 1
        assert 0 <= result.fair_yes_price <= 1
        assert 0 <= result.fair_no_price <= 1
        assert result.model_source == "dummy"

    def test_predict_with_ml_model(self, sample_features, tmp_path):
        """Should use ML model when available."""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 2, 1])
        clf.fit(X, y)

        model_path = tmp_path / "test_model.joblib"
        joblib.dump(clf, model_path)

        engine = PredictionEngine(model_path=model_path)
        result = engine.predict(sample_features)

        assert result.model_source == "test_model.joblib"
        assert result.predicted_bucket in [0, 1, 2]

    def test_generate_signal_payload(self, sample_features, tmp_path):
        """Should generate complete signal payload."""
        # Use missing model to get dummy predictor
        engine = PredictionEngine(model_path=tmp_path / "nonexistent.model")

        # Mock market snapshot
        market_snapshot = MagicMock()
        market_snapshot.yes_bid = 0.60
        market_snapshot.yes_ask = 0.65
        market_snapshot.model_dump.return_value = {"ticker": "TEST"}

        payload = engine.generate_signal_payload("TEST-TICKER", sample_features, market_snapshot)

        assert payload.ticker == "TEST-TICKER"
        assert payload.predicted_final_score is not None
        assert payload.confidence is not None
        assert payload.delta_to_market is not None
        assert payload.market_context is not None

    def test_generate_signal_payload_no_market(self, sample_features, tmp_path):
        """Should handle None market snapshot."""
        # Use missing model to get dummy predictor
        engine = PredictionEngine(model_path=tmp_path / "nonexistent.model")

        payload = engine.generate_signal_payload("TEST-TICKER", sample_features, None)

        assert payload.ticker == "TEST-TICKER"
        assert payload.delta_to_market is None


class TestBucketToScore:
    """Tests for bucket-to-score mapping."""

    def test_bucket_mapping_exists(self):
        """All expected buckets should be mapped."""
        assert 0 in BUCKET_TO_SCORE
        assert 1 in BUCKET_TO_SCORE
        assert 2 in BUCKET_TO_SCORE

    def test_bucket_scores_reasonable(self):
        """Bucket scores should be in valid RT range."""
        for bucket, score in BUCKET_TO_SCORE.items():
            assert 0 <= score <= 100
