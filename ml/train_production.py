"""
Fit the production inference path:

  raw 14 fields -> FeatureCleaner -> FeatureBuilder -> encode/scale -> XGBoost

Uses the same stratified 80/20 split as Phase 2 (random_state=42).
Does not overwrite dataset/processed/X_*.npy (those remain the Phase 2 26-col matrices).

Writes:
  models/preprocessor.pkl          clean + engineer + encode/scale
  models/final_model.pkl           XGBoost only
  models/inference_pipeline.pkl    preprocess Pipeline + XGBoost
  models/final_model_config.json
  models/production_feature_names.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    FINAL_MODEL_CONFIG_PATH,
    FINAL_MODEL_PATH,
    INFERENCE_PIPELINE_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    RAW_DATASET_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from ml.evaluate import recommend_threshold, threshold_table
from ml.feature_engineering.feature_builder import (
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
)
from ml.preprocess import IDENTIFIER_COLUMNS, REQUIRED_FEATURES, build_production_preprocessor

PRODUCTION_FEATURE_NAMES_PATH = MODELS_DIR / "production_feature_names.json"
PRODUCTION_METRICS_PATH = REPORTS_DIR / "production_fe_metrics.json"

# Phase 7 tuned XGBoost hyperparameters (GridSearchCV on roc_auc).
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.2,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _split_raw():
    raw = pd.read_csv(RAW_DATASET_PATH)
    feature_frame = raw.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in raw.columns])
    y = feature_frame[TARGET_COLUMN].astype(int)
    X = feature_frame.drop(columns=[TARGET_COLUMN])
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def engineered_row(preprocessor: Pipeline, raw_row: pd.DataFrame) -> dict:
    cleaned = preprocessor.named_steps["clean"].transform(raw_row)
    engineered = preprocessor.named_steps["engineer"].transform(cleaned)
    cols = ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
    values = engineered.loc[:, cols].iloc[0].to_dict()
    for key in ENGINEERED_NUMERIC_FEATURES:
        values[key] = float(values[key])
    for key in ENGINEERED_CATEGORICAL_FEATURES:
        values[key] = str(values[key])
    return values


def run_production_training() -> dict:
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(RAW_DATASET_PATH)

    X_train, X_test, y_train, y_test = _split_raw()

    preprocessor = build_production_preprocessor()
    preprocessor.fit(X_train, y_train)

    X_train_mat = preprocessor.transform(X_train)
    X_test_mat = preprocessor.transform(X_test)
    feature_names = preprocessor.named_steps["encode_scale"].get_feature_names_out().tolist()
    n_features = int(X_train_mat.shape[1])

    builder = preprocessor.named_steps["engineer"]
    cleaner = preprocessor.named_steps["clean"]

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_mat, y_train)

    train_proba = model.predict_proba(X_train_mat)[:, 1]
    test_proba = model.predict_proba(X_test_mat)[:, 1]
    test_pred_default = (test_proba >= 0.5).astype(int)

    thr_table = threshold_table(y_test.to_numpy(), test_proba)
    rec = recommend_threshold(thr_table)
    threshold = float(rec["recommended_threshold"])
    test_pred_op = (test_proba >= threshold).astype(int)

    metrics = {
        "n_model_features": n_features,
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
        "test_accuracy_at_0.5": float(accuracy_score(y_test, test_pred_default)),
        "test_precision_at_0.5": float(precision_score(y_test, test_pred_default, zero_division=0)),
        "test_recall_at_0.5": float(recall_score(y_test, test_pred_default, zero_division=0)),
        "test_f1_at_0.5": float(f1_score(y_test, test_pred_default, zero_division=0)),
        "test_accuracy_at_threshold": float(accuracy_score(y_test, test_pred_op)),
        "test_precision_at_threshold": float(rec["precision"]),
        "test_recall_at_threshold": float(rec["recall"]),
        "test_f1_at_threshold": float(rec["f1"]),
        "test_percent_flagged_at_threshold": float(rec["predicted_positive_rate"]),
        "threshold": threshold,
    }
    metrics["roc_auc_gap"] = metrics["train_roc_auc"] - metrics["test_roc_auc"]

    inference_pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(model, FINAL_MODEL_PATH)
    joblib.dump(inference_pipeline, INFERENCE_PIPELINE_PATH)
    PRODUCTION_FEATURE_NAMES_PATH.write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )

    config = {
        "model": "XGBoost",
        "includes_feature_engineering": True,
        "n_model_features": n_features,
        "preprocessor": PREPROCESSOR_PATH.name,
        "final_model": FINAL_MODEL_PATH.name,
        "inference_pipeline": INFERENCE_PIPELINE_PATH.name,
        "pairing": (
            "Preferred: load inference_pipeline.pkl and call predict_proba on the "
            "raw 14-field DataFrame. Equivalent: preprocessor.pkl then final_model.pkl. "
            "Do not refit. Do not send engineered fields from the client."
        ),
        "pipeline_steps": ["clean", "engineer", "encode_scale", "model"],
        "raw_input_fields": list(REQUIRED_FEATURES),
        "engineered_features": ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES,
        "price_q33": float(builder.price_q33_),
        "price_q66": float(builder.price_q66_),
        "product_price_cap": float(cleaner.price_cap_),
        "best_params": {
            "n_estimators": XGB_PARAMS["n_estimators"],
            "max_depth": XGB_PARAMS["max_depth"],
            "learning_rate": XGB_PARAMS["learning_rate"],
        },
        "threshold": threshold,
        "threshold_rule": rec["rule"],
        "default_predict_label": "returned if predict_proba[:, 1] >= threshold",
        **{k: v for k, v in metrics.items() if k != "threshold"},
    }
    FINAL_MODEL_CONFIG_PATH.write_text(
        json.dumps(_json_ready(config), indent=2), encoding="utf-8"
    )
    PRODUCTION_METRICS_PATH.write_text(
        json.dumps(_json_ready({"metrics": metrics, "threshold_table": thr_table.to_dict(orient="records")}), indent=2),
        encoding="utf-8",
    )

    sample = X_test.iloc[[0]][REQUIRED_FEATURES]
    smoke_engineered = engineered_row(preprocessor, sample)
    smoke_mat = preprocessor.transform(sample)
    if smoke_mat.shape[1] != n_features:
        raise RuntimeError(f"Expected {n_features} columns, got {smoke_mat.shape[1]}")
    pipe_proba = float(inference_pipeline.predict_proba(sample)[0, 1])
    split_proba = float(model.predict_proba(smoke_mat)[0, 1])
    if abs(pipe_proba - split_proba) > 1e-9:
        raise RuntimeError("inference_pipeline.pkl disagrees with preprocessor + model")

    smoke = {
        "raw_input": json.loads(sample.iloc[0].to_json()),
        "engineered_features": smoke_engineered,
        "n_model_features": n_features,
        "probability": pipe_proba,
        "flagged": pipe_proba >= threshold,
        "ok": True,
    }

    print("Production pipeline fitted.")
    print(f"  n_features={n_features}  test_roc_auc={metrics['test_roc_auc']:.6f}  gap={metrics['roc_auc_gap']:.6f}")
    print(f"  threshold={threshold}  precision={rec['precision']:.4f}  recall={rec['recall']:.4f}")
    print(f"  {PREPROCESSOR_PATH}")
    print(f"  {FINAL_MODEL_PATH}")
    print(f"  {INFERENCE_PIPELINE_PATH}")
    print("  smoke engineered", json.dumps(smoke_engineered))
    print(f"  smoke probability={pipe_proba:.6f}")
    return {
        "metrics": metrics,
        "config": config,
        "feature_names": feature_names,
        "smoke": smoke,
        "threshold_table": thr_table,
        "recommendation": rec,
    }


def main() -> None:
    run_production_training()


if __name__ == "__main__":
    main()
