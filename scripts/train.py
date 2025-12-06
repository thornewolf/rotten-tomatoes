import logging

import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_rotten_tomatoes_model(data_path: str):
    """
    Skeleton for training a Random Forest on scraped data.
    """
    logger.info("Loading data from %s...", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows", len(df))

    feature_cols = ["days_since_release", "current_rating", "num_reviews"]
    target_col = "final_rating_bucket"

    missing_columns = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")

    # Basic tabular model using the numerical columns as-is.
    X = df[feature_cols]
    y = df[target_col]

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info("Validation accuracy: %.3f", acc)

    model_path = Path("prediction.model")
    joblib.dump(clf, model_path)
    logger.info("Saved model to %s", model_path.resolve())

    return {"accuracy": acc, "n_rows": len(df), "model_path": str(model_path)}


if __name__ == "__main__":
    train_rotten_tomatoes_model("data/processed.csv")
