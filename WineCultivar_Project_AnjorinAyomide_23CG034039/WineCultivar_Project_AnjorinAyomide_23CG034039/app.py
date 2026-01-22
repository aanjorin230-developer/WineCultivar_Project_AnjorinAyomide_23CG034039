"""Flask web app for Wine cultivar classification.

Run locally:
    python app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from flask import Flask, render_template, request, url_for
from sklearn.datasets import load_wine


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "wine_cultivar_model.pkl"

# Project requirement: use exactly six input features from the approved list.
FEATURE_COLUMNS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "flavanoids",
    "od280/od315_of_diluted_wines",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    min_val: float
    max_val: float
    default: float
    step: float


app = Flask(__name__)


@lru_cache(maxsize=1)
def load_model() -> object:
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Expected at: "
            f"{MODEL_PATH}. Please train or place the model there."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_specs() -> Tuple[List[FeatureSpec], Dict[int, str]]:
    """Build feature metadata and label mapping using the Wine dataset."""
    dataset = load_wine()
    data = dataset.data
    feature_names = dataset.feature_names
    target_names = {idx: f"Cultivar {idx + 1}" for idx in range(len(dataset.target_names))}

    specs: List[FeatureSpec] = []
    for feature in FEATURE_COLUMNS:
        idx = feature_names.index(feature)
        values = data[:, idx]
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        default = float(np.mean(values))
        step = max((max_val - min_val) / 100, 0.001)
        specs.append(FeatureSpec(feature, min_val, max_val, default, step))

    return specs, target_names


def parse_float(value: str, spec: FeatureSpec) -> Tuple[float, str | None]:
    """Parse and validate a float from form data."""
    if value is None or value.strip() == "":
        return 0.0, f"{spec.name.replace('_', ' ').title()} is required."
    try:
        number = float(value)
    except ValueError:
        return 0.0, f"{spec.name.replace('_', ' ').title()} must be a number."
    if number < spec.min_val or number > spec.max_val:
        return (
            number,
            f"{spec.name.replace('_', ' ').title()} must be between {spec.min_val:.4f} and {spec.max_val:.4f}.",
        )
    return number, None


@app.route("/", methods=["GET", "POST"])
def index():
    specs, target_names = load_feature_specs()
    values: Dict[str, float] = {spec.name: spec.default for spec in specs}
    errors: List[str] = []
    prediction_label: str | None = None
    model_error: str | None = None

    if request.method == "POST":
        for spec in specs:
            raw_value = request.form.get(spec.name)
            parsed_value, error = parse_float(raw_value, spec)
            values[spec.name] = parsed_value if error is None else values[spec.name]
            if error:
                errors.append(error)

        if not errors:
            try:
                model = load_model()
                input_array = np.array([values[spec.name] for spec in specs], dtype=float).reshape(1, -1)
                prediction = int(model.predict(input_array)[0])
                prediction_label = target_names.get(prediction, f"Cultivar {prediction + 1}")
            except FileNotFoundError as exc:
                model_error = str(exc)

    return render_template(
        "index.html",
        specs=specs,
        values=values,
        errors=errors,
        prediction=prediction_label,
        model_error=model_error,
    )


if __name__ == "__main__":
    app.run(debug=False)
