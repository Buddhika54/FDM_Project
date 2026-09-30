"""
Phase 7 — select a production model and operating threshold.

Does not re-run GridSearchCV. Uses tuned pickles + Phase 2 test arrays.
Saves models/final_model.pkl (estimator) to pair with models/preprocessor.pkl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    FINAL_MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATASET_PATH,
    REPORTS_DIR,
)
from ml.evaluate import (
    THRESHOLDS,
    plot_threshold_curves,
    recommend_threshold,
    threshold_table,
)
from ml.preprocess import REQUIRED_FEATURES
from ml.tune import PARAMS_OUT, TUNED_PATHS

CONFIG_PATH = MODELS_DIR / "final_model_config.json"
SUMMARY_PATH = REPORTS_DIR / "final_model_summary.md"
XGB_THR_CSV = REPORTS_DIR / "tuned_xgboost_threshold_analysis.csv"
XGB_THR_FIG = REPORTS_DIR / "figures" / "16_tuned_xgboost_threshold.png"
SELECTED_NAME = "XGBoost"
SELECTED_SOURCE = TUNED_PATHS[SELECTED_NAME]


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


def load_tuned_comparison(path: Path = REPORTS_DIR / "tuned_model_comparison.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Need Phase 6 table: {path}")
    return pd.read_csv(path)


def xgboost_threshold_sweep(processed_dir: Path = PROCESSED_DATA_DIR) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    model = joblib.load(SELECTED_SOURCE)
    proba = model.predict_proba(X_test)[:, 1]
    table = threshold_table(y_test, proba)
    return table, y_test, proba


def write_summary(row: pd.Series, rec: dict, params: dict) -> str:
    flagged_pct = 100.0 * rec["predicted_positive_rate"]
    text = (
        f"The production estimator is **tuned XGBoost** "
        f"(`n_estimators={params.get('n_estimators')}`, "
        f"`max_depth={params.get('max_depth')}`, "
        f"`learning_rate={params.get('learning_rate')}`), "
        f"paired with the Phase 2 `preprocessor.pkl`. "
        f"Test ROC-AUC is {row['roc_auc']:.3f} — on the same ~0.59 plateau as LR ({0.590:.3f}) "
        f"and RF ({0.594:.3f}); that is a ranking ceiling from weak numeric signal, not a high-accuracy detector. "
        f"XGBoost is chosen over RF because the train–test gap is {row['roc_auc_gap']:.3f} (vs RF 0.071) "
        f"and over LR because recall-first flagging needs a slightly better ranker with a closed gap, "
        f"while LR remains the calibration reference. "
        f"The operating threshold is **{rec['recommended_threshold']:.1f}** (not 0.5): "
        f"hold-out precision {rec['precision']:.1%}, recall {rec['recall']:.1%}, "
        f"flagging about {flagged_pct:.0f}% of orders. "
        f"Use these figures as expected production behaviour; do not advertise accuracy well above 60%."
    )
    return text


def smoke_test_raw_rows(n: int = 3) -> dict:
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(PREPROCESSOR_PATH)
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(FINAL_MODEL_PATH)
    raw = pd.read_csv(RAW_DATASET_PATH)
    sample = raw.loc[:, REQUIRED_FEATURES].head(n)
    pre = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(FINAL_MODEL_PATH)
    transformed = pre.transform(sample)
    proba = model.predict_proba(transformed)[:, 1]
    if transformed.shape[0] != n:
        raise RuntimeError("Preprocessor dropped rows")
    n_features = int(transformed.shape[1])
    expected = None
    if CONFIG_PATH.exists():
        expected = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("n_model_features")
    if expected is not None and n_features != int(expected):
        raise RuntimeError(f"Expected {expected} model features, got {n_features}")
    return {
        "n_rows": n,
        "transformed_shape": list(transformed.shape),
        "proba": [float(x) for x in proba],
        "ok": True,
    }


def run_selection() -> dict:
    if CONFIG_PATH.exists():
        existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if existing.get("includes_feature_engineering"):
            raise RuntimeError(
                "Production artifacts include FeatureBuilder. "
                "Re-run `python -m ml.train_production` instead of Phase 7 select_model."
            )
    comparison = load_tuned_comparison()
    params = json.loads(PARAMS_OUT.read_text(encoding="utf-8"))
    xgb_row = comparison.set_index("model").loc[SELECTED_NAME]

    table, y_test, proba = xgboost_threshold_sweep()
    rec = recommend_threshold(table)
    REPORTS_DIR.joinpath("figures").mkdir(parents=True, exist_ok=True)
    table.to_csv(XGB_THR_CSV, index=False)
    plot_threshold_curves(y_test, proba, XGB_THR_FIG, rec["recommended_threshold"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(SELECTED_SOURCE)
    joblib.dump(model, FINAL_MODEL_PATH)

    config = {
        "model": SELECTED_NAME,
        "source_pickle": SELECTED_SOURCE.name,
        "preprocessor": PREPROCESSOR_PATH.name,
        "pairing": "Load preprocessor.pkl then final_model.pkl; do not refit either.",
        "threshold": rec["recommended_threshold"],
        "test_roc_auc": float(xgb_row["roc_auc"]),
        "train_roc_auc": float(xgb_row["train_roc_auc"]),
        "roc_auc_gap": float(xgb_row["roc_auc_gap"]),
        "cv_roc_auc": float(xgb_row["cv_roc_auc"]),
        "best_params": params[SELECTED_NAME],
        "test_recall_at_threshold": rec["recall"],
        "test_precision_at_threshold": rec["precision"],
        "test_f1_at_threshold": rec["f1"],
        "test_percent_flagged_at_threshold": rec["predicted_positive_rate"],
        "threshold_rule": rec["rule"],
        "default_predict_label": "returned if predict_proba[:, 1] >= threshold",
    }
    CONFIG_PATH.write_text(json.dumps(_json_ready(config), indent=2), encoding="utf-8")

    summary = write_summary(xgb_row, rec, params[SELECTED_NAME])
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")

    smoke = smoke_test_raw_rows()

    print("Phase 7 complete (no Flask API).")
    print(f"  selected: {SELECTED_NAME}  threshold={rec['recommended_threshold']}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"  at {rec['recommended_threshold']:.1f}: "
        f"precision={rec['precision']:.4f} recall={rec['recall']:.4f} "
        f"flagged={rec['predicted_positive_rate']:.4f}"
    )
    print("  smoke test", smoke)
    print(f"  {FINAL_MODEL_PATH}")
    print(f"  {CONFIG_PATH}")
    print(f"  {SUMMARY_PATH}")
    return {
        "comparison": comparison,
        "xgboost_thresholds": table,
        "recommendation": rec,
        "config": config,
        "summary": summary,
        "smoke": smoke,
        "params": params,
    }


def main() -> None:
    run_selection()


if __name__ == "__main__":
    main()
