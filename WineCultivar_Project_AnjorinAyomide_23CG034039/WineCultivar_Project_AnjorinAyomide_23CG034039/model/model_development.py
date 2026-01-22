"""Model development script for Wine Cultivar Origin Prediction System.

This script follows Project 6 requirements:
- Uses exactly six approved features from the Wine dataset.
- Handles missing values, scales features, and trains a multiclass classifier.
- Evaluates with accuracy, precision, recall, F1 (macro), and classification report.
- Saves the trained model to /model/wine_cultivar_model.pkl.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# Approved feature list (must use exactly six inputs).
FEATURE_COLUMNS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "flavanoids",
    "od280/od315_of_diluted_wines",
]


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load the Wine dataset and return selected features and target."""
    dataset = load_wine()
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    X = data[FEATURE_COLUMNS].copy()
    y = pd.Series(dataset.target, name="cultivar")
    target_names = [f"Cultivar {idx + 1}" for idx in range(len(dataset.target_names))]
    return X, y, target_names


def build_pipeline() -> Pipeline:
    """Build preprocessing + classifier pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, target_names: List[str]) -> None:
    """Evaluate model performance and print metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )

    print("\nEvaluation Metrics")
    print("------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))


def save_model(model: Pipeline, output_path: Path) -> None:
    """Save trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def main() -> None:
    """Train and persist the model."""
    X, y, target_names = load_dataset()

    # Handle missing values (if any) and ensure stratified split for class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_test, y_test, target_names)

    model_path = Path("model/wine_cultivar_model.pkl")
    save_model(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
