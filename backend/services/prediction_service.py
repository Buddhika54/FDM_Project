"""
Inference use-case. Only module that calls sklearn/xgboost pickles.
"""

from __future__ import annotations

import pandas as pd

from models.prediction_schema import FEATURE_COLUMN_ORDER
from services.model_loader import get_model, get_model_config, get_preprocessor


def probability_to_risk_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Medium"
    return "High"


def recommendation_for(flagged: bool, risk_level: str) -> str:
    if flagged:
        return (
            "Flag for pre-fulfillment review (recall-first threshold). "
            "Confirm the order with the customer before shipping."
        )
    if risk_level == "Low":
        return "Low predicted return risk. Proceed with standard fulfillment."
    return "Below the intervention threshold. Monitor; no automatic hold."


def predict_return_risk(features: dict) -> dict:
    frame = pd.DataFrame([features], columns=FEATURE_COLUMN_ORDER)
    preprocessor = get_preprocessor()
    model = get_model()
    meta = get_model_config()
    threshold = float(meta["threshold"])

    transformed = preprocessor.transform(frame)
    probability = float(model.predict_proba(transformed)[0, 1])
    flagged = bool(probability >= threshold)
    prediction = 1 if flagged else 0
    risk_level = probability_to_risk_level(probability)
    params = meta.get("best_params") or {}
    version = (
        f"tuned-n_estimators={params.get('n_estimators')}"
        f"-max_depth={params.get('max_depth')}"
        f"-lr={params.get('learning_rate')}"
        if params
        else "phase7"
    )
    return {
        "return_risk_probability": probability,
        "flagged": flagged,
        "model": meta.get("model", "XGBoost"),
        "model_version": version,
        "threshold": threshold,
        "prediction": prediction,
        "prediction_label": "Returned" if flagged else "Not Returned",
        "risk_level": risk_level,
        "probability": probability,
        "probability_percent": round(probability * 100.0, 2),
        "recommendation": recommendation_for(flagged, risk_level),
    }
