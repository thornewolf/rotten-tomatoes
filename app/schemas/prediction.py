"""
Pydantic models for prediction service data structures.
Provides type safety for ML model inputs and outputs.
"""

from pydantic import BaseModel, Field


class PredictionFeatures(BaseModel):
    """Features used for Rotten Tomatoes score prediction."""

    days_since_release: float | None = Field(None, ge=0, description="Days since theatrical release")
    current_rating: float | None = Field(None, ge=0, le=100, description="Current Rotten Tomatoes rating (0-100)")
    num_reviews: float | None = Field(None, ge=0, description="Number of critic reviews on Rotten Tomatoes")

    class Config:
        json_schema_extra = {
            "example": {
                "days_since_release": 7.0,
                "current_rating": 72.5,
                "num_reviews": 125.0,
            }
        }


class PredictionResult(BaseModel):
    """Result from the prediction engine."""

    predicted_bucket: int = Field(..., description="Predicted rating bucket (0=low, 1=mid, 2=high)")
    predicted_final_score: float = Field(..., ge=0, le=100, description="Expected final RT score")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence in prediction")
    probabilities: dict[str, float] = Field(..., description="Probability distribution over buckets")
    fair_yes_price: float = Field(..., ge=0, le=1, description="Fair price for YES side")
    fair_no_price: float = Field(..., ge=0, le=1, description="Fair price for NO side")
    features: dict[str, float] = Field(..., description="Normalized feature values used")
    defaults_used: list[str] = Field(default_factory=list, description="Features that used default values")
    model_source: str = Field(..., description="Model used ('dummy' or model filename)")


class RTContext(BaseModel):
    """Rotten Tomatoes context for signal response."""

    features_used: dict[str, float] = Field(..., description="Feature values used in prediction")
    defaults_used: list[str] = Field(default_factory=list, description="Features that defaulted")
    predicted_bucket: int = Field(..., description="Predicted rating bucket")


class SignalResponse(BaseModel):
    """Complete signal response with prediction and market context."""

    ticker: str = Field(..., description="Market ticker")
    predicted_final_score: float = Field(..., ge=0, le=100, description="Predicted final RT score")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    fair_yes_price: float = Field(..., ge=0, le=1, description="Model's fair YES price")
    fair_no_price: float = Field(..., ge=0, le=1, description="Model's fair NO price")
    delta_to_market: float | None = Field(None, description="Difference between fair and market price")
    probabilities: dict[str, float] = Field(..., description="Probability distribution")
    model_source: str = Field(..., description="Model used for prediction")
    rt_context: RTContext = Field(..., description="Rotten Tomatoes prediction context")
    market_context: dict = Field(default_factory=dict, description="Market data context")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "ROTTOM-23DEC-T75",
                "predicted_final_score": 78.5,
                "confidence": 0.72,
                "fair_yes_price": 0.68,
                "fair_no_price": 0.32,
                "delta_to_market": 0.05,
                "probabilities": {"0": 0.15, "1": 0.28, "2": 0.57},
                "model_source": "prediction.model",
                "rt_context": {
                    "features_used": {"days_since_release": 7.0, "current_rating": 72.5, "num_reviews": 125.0},
                    "defaults_used": [],
                    "predicted_bucket": 2,
                },
                "market_context": {"market_yes_mid_price": 0.63},
            }
        }
