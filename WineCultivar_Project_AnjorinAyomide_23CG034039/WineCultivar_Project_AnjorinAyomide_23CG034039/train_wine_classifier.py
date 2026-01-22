"""Wine multiclass classification pipeline.

This script loads the Wine dataset, preprocesses features, performs feature selection,
trains a multiclass classifier, evaluates it, and saves the trained model to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the Wine dataset into a DataFrame and Series."""
    data = load_wine()
    features = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.Series(data.target, name="target")
    return features, target, data.target_names.tolist()


def handle_missing_values(features: pd.DataFrame) -> pd.DataFrame:
    """Ensure missing values are handled (imputation happens in pipeline)."""
    if features.isna().any().any():
        return features.copy()
    return features


def build_pipeline(num_features: int) -> Pipeline:
    """Build preprocessing + feature selection + classifier pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "feature_selection",
                SelectKBest(score_func=mutual_info_classif, k=min(10, num_features)),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_hyperparameters(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Tune SVM hyperparameters using GridSearchCV."""
    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__gamma": ["scale", 0.01, 0.1, 1.0],
        "feature_selection__k": [8, 10, "all"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, target_names: list[str]) -> dict:
    """Evaluate model and return metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    report = classification_report(y_test, predictions, target_names=target_names, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": report,
    }


def save_model(model, output_path: Path) -> None:
    """Save trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def main() -> None:
    """Run training, evaluation, and persistence."""
    features, target, target_names = load_dataset()
    features = handle_missing_values(features)

    class_counts = target.value_counts().to_dict()
    print("Class distribution:", class_counts)
    if len(set(class_counts.values())) != 1:
        print("Note: Class distribution is uneven; class_weight='balanced' is enabled.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipeline = build_pipeline(num_features=features.shape[1])
    tuned_model = tune_hyperparameters(pipeline, X_train, y_train)

    metrics = evaluate_model(tuned_model, X_test, y_test, target_names)

    print("\nEvaluation Metrics")
    print("------------------")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    output_path = Path("model/wine_svm_pipeline.joblib")
    save_model(tuned_model, output_path)

    metadata = {
        "model_path": str(output_path),
        "best_params": tuned_model.best_params_,
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
    }
    metadata_path = Path("model/wine_svm_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nModel saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
